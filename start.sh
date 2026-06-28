#!/usr/bin/env bash
set -e

for var in REDIRECT_URI COOKIE_SECRET GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET; do
  if [ -z "${!var}" ]; then
    echo "ERROR: $var is not set. Add it in Render → Environment before deploying." >&2
    exit 1
  fi
done

mkdir -p .streamlit
cat > .streamlit/secrets.toml <<EOF
[auth]
redirect_uri = "${REDIRECT_URI}"
cookie_secret = "${COOKIE_SECRET}"
client_id = "${GOOGLE_CLIENT_ID}"
client_secret = "${GOOGLE_CLIENT_SECRET}"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
EOF

streamlit run streamlit_app.py --server.port "$PORT" --server.address 0.0.0.0
