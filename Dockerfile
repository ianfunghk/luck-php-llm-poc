# php-llm Dockerfile.
#
# Builds a single image that contains BOTH the PHP-FPM runtime and Nginx,
# ready to be used either standalone (with `docker run -p 8080:80 php-llm`)
# or wired up by docker-compose.yml (recommended, since this project needs
# the `weights/` directory mounted in from the host).
#
# Layout inside the image:
#   /var/www/html/        PHP source (infer.php, src/, web/)
#   /var/www/html/weights/  empty placeholder, expected to be volume-mounted
#   /usr/local/etc/php/conf.d/99-php-llm.ini   custom php.ini
#   /etc/nginx/conf.d/default.conf             nginx server block
#
# supervisor runs php-fpm + nginx in one container. This image is "PHP runtime
# only" per the user's choice — no Python is installed here.

FROM php:8.2-fpm-alpine AS runtime

# Install nginx + supervisor + the shmop extension (bundled; just needs enabling
# for the WeightCache shared-memory fast path).
RUN apk add --no-cache nginx supervisor && \
    docker-php-ext-install opcache shmop && \
    # Nginx needs a place for its PID/logs.
    mkdir -p /var/run/nginx /var/log/nginx && \
    # App directory; weights/ is a placeholder for a volume mount.
    mkdir -p /var/www/html/weights && \
    chown -R www-data:www-data /var/www/html

# Custom php.ini with memory_limit=1024M and max_execution_time=120.
COPY php.ini /usr/local/etc/php/conf.d/99-php-llm.ini

# Nginx server block.
COPY nginx.conf /etc/nginx/http.d/default.conf

# Supervisor program list: one php-fpm pool + one nginx master.
COPY <<'EOF' /etc/supervisor.d/php-llm.ini
[program:php-fpm]
command=php-fpm --nodaemonize --force-stderr
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:nginx]
command=nginx -g 'daemon off;'
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
EOF

# Tell nginx-alpine where its main config lives (matches nginx.conf above).
ENV NGINX_USER=nginx

# Application code.
COPY infer.php /var/www/html/infer.php
COPY src/      /var/www/html/src/
COPY web/      /var/www/html/web/

EXPOSE 80
WORKDIR /var/www/html

# Default entry: supervisor brings up php-fpm + nginx.
CMD ["supervisord", "-n", "-c", "/etc/supervisord.conf"]
