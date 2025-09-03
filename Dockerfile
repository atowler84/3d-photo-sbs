# syntax=docker/dockerfile:1.6

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl ca-certificates \
    libgl1 libglib2.0-0 libegl1 libgles2 libglvnd0 libxext6 libx11-6 libxrender1 libxi6 libsm6 \
    libgl1-mesa-dri mesa-utils \
    xvfb xauth dumb-init \
  && rm -rf /var/lib/apt/lists/*


RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} -s /bin/bash app

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi && \
    pip install --no-cache-dir PyQt5 "vispy>=0.14" PyOpenGL PyOpenGL_accelerate

# App code (own it as non-root so it can write)
COPY --chown=${UID}:${GID} . .

# Writable dirs
RUN install -d -o ${UID} -g ${GID} \
    /home/app/.cache/huggingface /home/app/.config/matplotlib \
    /home/app/.cache/mesa_shader_cache /tmp/runtime-app \
    /app/depth /app/checkpoints /app/video /app/image /app/mesh

# Ensure system libstdc++/libgcc are used when GLX starts
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 \
    XDG_RUNTIME_DIR=/tmp/runtime-app

# Headless GLX on virtual display; CUDA is independent for ML
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface \
    MPLCONFIGDIR=/home/app/.config/matplotlib \
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen \
    VISPY_APP_BACKEND=pyqt5 \
    VISPY_USE_APP=pyqt5 \
    PYOPENGL_PLATFORM=glx \
    LIBGL_ALWAYS_SOFTWARE=1 \
    # ensure swrast (software GL) is found when GLX starts
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

EXPOSE 8008
HEALTHCHECK --interval=30s --timeout=10s --retries=5 CMD curl -fsS http://127.0.0.1:8008/api/health || exit 1

# Tiny entrypoint that brings up Xvfb then starts the server
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER app
ENTRYPOINT ["dumb-init","--","/usr/local/bin/entrypoint.sh"]
