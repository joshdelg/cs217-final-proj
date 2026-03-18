#!/usr/bin/env bash
set -euo pipefail

# ---------- Install InfluxDB 2.x ----------
# Add the InfluxData key to verify downloads and add the repository
curl --silent --location -O https://repos.influxdata.com/influxdata-archive.key
gpg --show-keys --with-fingerprint --with-colons ./influxdata-archive.key 2>&1 \
| grep -q '^fpr:\+24C975CBA61A024EE1B631787C3D57159FC2F927:$' \
&& cat influxdata-archive.key \
| gpg --dearmor \
| sudo tee /etc/apt/keyrings/influxdata-archive.gpg > /dev/null \
&& echo 'deb [signed-by=/etc/apt/keyrings/influxdata-archive.gpg] https://repos.influxdata.com/debian stable main' \
| sudo tee /etc/apt/sources.list.d/influxdata.list

sudo apt-get update && sudo apt-get install -y influxdb2
sudo service influxdb start

# Wait for InfluxDB to be ready
echo "Waiting for InfluxDB to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8086/health | grep -q '"status":"pass"'; then
        echo "InfluxDB is ready."
        break
    fi
    sleep 1
done

# ---------- Initial setup (matches profiler_influx.json) ----------
# username: admin / password: adminadmin / org: joshdelg-cs217 / bucket: cs217
influx setup \
    --username admin \
    --password adminadmin \
    --org joshdelg-cs217 \
    --bucket cs217 \
    --retention 0 \
    --force

echo ""
echo "InfluxDB setup complete."
echo "  endpoint : http://localhost:8086"
echo "  org      : joshdelg-cs217"
echo "  bucket   : cs217"
echo "  user     : admin"