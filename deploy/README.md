# Despliegue del proyecto en servidor (DigitalOcean Droplet)

Provisioning bare-metal para servir la app Streamlit de triaje-ia-tfg
con Ollama local, Caddy (HTTPS automático) y DuckDNS como dominio
público.

## Requisitos previos

- Cuenta en DigitalOcean con créditos y `doctl` instalado y autenticado
  (`doctl auth init`).
- Subdominio `triaje.duckdns.org` reclamado en https://www.duckdns.org
  con su token asociado.
- Clave SSH cargada en DigitalOcean.

## 1. Crear Cloud Firewall y Droplet

```bash
# Firewall independiente (reutilizable). SSH restringido a tu IP.
doctl compute firewall create \
  --name triaje-fw \
  --inbound-rules "protocol:tcp,address:84.121.57.78/32,port:22 protocol:tcp,address:0.0.0.0/0,port:80 protocol:tcp,address:0.0.0.0/0,port:443" \
  --outbound-rules "protocol:tcp,address:0.0.0.0/0,port:all protocol:udp,address:0.0.0.0/0,port:all protocol:icmp,address:0.0.0.0/0"

# Droplet 16 GB en Frankfurt.
doctl compute droplet create triaje-setup \
  --region fra1 \
  --size s-8vcpu-16gb \
  --image ubuntu-24-04-x64 \
  --ssh-keys <tu-key-id> \
  --firewall-id <firewall-id>
```

## 2. Provisioning del Droplet

```bash
ssh root@<IP>
git clone https://github.com/<usuario>/triaje-ia-tfg.git
cd triaje-ia-tfg
DUCKDNS_TOKEN=<token> bash deploy/setup.sh
```

Si Caddy pide el hash de basicauth:

```bash
caddy hash-password           # introduce la contraseña deseada
nano /etc/caddy/Caddyfile     # pega el hash donde está <PEGAR_HASH_AQUI>
systemctl reload caddy
```

## 3. Actualizaciones del código

Tras editar y pushear en tu PC:

```bash
ssh root@<IP>
cd ~/triaje-ia-tfg
bash deploy/update.sh
```

## 4. Snapshot de seguridad

Cuando la app funcione correctamente, crear un snapshot como backup:

```bash
doctl compute droplet-action snapshot <droplet-id> --snapshot-name triaje-backup-$(date +%Y%m%d)
doctl compute snapshot list
```

Anotar el `snapshot-id` aquí:

- snapshot-id: __PENDIENTE__

El snapshot permite reconstruir el droplet en minutos si se rompe,
conservando servicios instalados, certificado Caddy y modelo Ollama
descargado.

## 5. Cierre (tras la defensa)

```bash
doctl compute droplet delete <droplet-id>
doctl compute snapshot delete <snapshot-id>     # opcional
doctl compute firewall delete <firewall-id>     # opcional
```

## Notas técnicas

- **Swap 4 GB**: red de seguridad anti-OOM si el modelo LLM y Streamlit
  coinciden en un pico de memoria.
- **unattended-upgrades**: el reboot automático está desactivado para
  evitar un reinicio inesperado durante la semana de la defensa.
- **ollama-warmup.service**: precarga el modelo en RAM en cada arranque
  para eliminar el cold start de la primera inferencia.
- **Firewall**: el puerto 11434 de Ollama no está expuesto públicamente;
  solo accesible en localhost.
- **Token DuckDNS**: se inyecta como variable de entorno al ejecutar
  `setup.sh`; nunca se commitea en el repositorio.
- **Certificado Caddy**: Let's Encrypt lo gestiona automáticamente. El
  snapshot lo conserva, por lo que restaurar un droplet no dispara un
  nuevo challenge ni riesgo de rate limit.
