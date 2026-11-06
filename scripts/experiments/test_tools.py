"""CPU integration checks for scoring, resume and configuration isolation.

Uses a fake HTTP generation server; this does not validate GPU accuracy/latency.
Run with the data environment: python -m unittest discover -s scripts/experiments.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

REPO = Path(__file__).resolve().parents[2]


class ExperimentToolsTest(unittest.TestCase):
    def setUp(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root/'model'
        self.model.mkdir()
        tokenizer = Tokenizer(models.WordLevel({'[UNK]':0}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(self.model/'tokenizer.json'))
        self.requests = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"test_server":true}')

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'text':'The correct answer is A. ALPHA', 'meta_info':{'completion_tokens':8}}).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def command(self, dataset, data, output):
        return [sys.executable, str(REPO/'scripts/experiments/run_experiment.py'),
                '--method','scale','--dataset',dataset,'--data',str(data),'--model',str(self.model),
                '--output',str(output),'--server',self.url,'--resume']

    def run_checked(self, command):
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)

    def test_ruler_resume_and_config_isolation(self):
        data = self.root/'ruler'
        path = data/'32k/niah_single_1/validation.jsonl'
        path.parent.mkdir(parents=True)
        records = [{'index':i, 'input':'Find ALPHA', 'outputs':['ALPHA'], 'length':32768} for i in range(2)]
        path.write_text(''.join(json.dumps(row)+'\n' for row in records))
        output = self.root/'ruler-results'
        command = self.command('ruler', data, output)+['--lengths','32k']
        self.run_checked(command)
        report = json.loads((output/'summary.json').read_text())
        self.assertEqual((report['examples'],report['successful'],report['accuracy']), (2,2,1.0))
        self.assertEqual(len(self.requests), 2)
        self.run_checked(command)
        self.assertEqual(len(self.requests), 2, 'Resume must not regenerate successful examples')
        result = subprocess.run(command+['--dense-final','1'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.requests), 2)
        path.write_text(path.read_text()+json.dumps({'index':2,'input':'different data','outputs':['B']})+'\n')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, 'Changed data must not be appended to the same run')

    def test_longbench_scoring_and_resume(self):
        row = {'_id':'one','domain':'test','sub_domain':'test','difficulty':'easy','length':'short',
               'question':'Select A','context':'Example context','choice_A':'yes','choice_B':'no',
               'choice_C':'no','choice_D':'no','answer':'A'}
        data = self.root/'longbench.json'
        data.write_text(json.dumps([row, {**row,'_id':'two'}]))
        output = self.root/'longbench-results'
        command = self.command('longbench-v2', data, output)
        self.run_checked(command)
        self.run_checked(command)
        report = json.loads((output/'summary.json').read_text())
        self.assertEqual((report['examples'],report['successful'],report['accuracy']), (2,2,1.0))
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(r['sampling_params']['temperature'] == 0 for r in self.requests))


if __name__ == '__main__':
    unittest.main()
