# Exp50 Status

Current status: `VOID_WEIGHT_DOWNLOAD_BLOCKED`.

B0 permission recovery passed and B1 official VOID repo clone/audit passed. B2 attempted official HuggingFace downloads for `netflix/void-model` and `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`, but both failed from PAI with `httpx.ConnectError: [Errno 101] Network is unreachable`.

No fallback, mirror, or fabricated asset was used. Env smoke, trainable-forward audit, quadmask Gate8, and inference smoke remain blocked until the exact weights/base model are available.
