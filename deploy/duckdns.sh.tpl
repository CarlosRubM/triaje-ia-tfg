#!/usr/bin/env bash
# Plantilla DuckDNS — el token se sustituye por envsubst durante setup.sh.
# NO poner el token real aquí. El token se inyecta vía DUCKDNS_TOKEN.
echo url="https://www.duckdns.org/update?domains=triaje&token=${DUCKDNS_TOKEN}&ip=" \
  | curl -k -o /var/log/duckdns.log -K -
