# Reconstructs the dependency changes made in the running H20 fine-block host.
# The complete Docker build has not been rerun as part of this code publication.
FROM slimerl/slime@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f

ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python -m pip uninstall -y flashinfer-jit-cache
RUN python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu130
RUN python -m pip install --index-url ${PIP_INDEX_URL} \
      sglang-kernel==0.4.2.post1 sgl-deep-gemm==0.1.4.post1 \
      flashinfer-python==0.6.13 flashinfer-cubin==0.6.13 \
      transformers==5.12.1 numpy==1.26.4 tilelang==0.1.11 apache-tvm-ffi==0.1.11 \
      'cuda-toolkit[nvcc,cccl,crt,nvvm]==13.0.2'
COPY scripts/env/fix_cuda_layout.py /opt/scale/fix_cuda_layout.py
RUN python /opt/scale/fix_cuda_layout.py
ENV CUDA_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
ENV PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV MAX_JOBS=4
WORKDIR /workspace/Scale
CMD ["bash"]
