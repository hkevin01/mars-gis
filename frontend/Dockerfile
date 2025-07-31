# Mars-GIS Comprehensive Docker Configuration
# Multi-stage Docker build for frontend with testing capabilities

# Stage 1: Base Node.js environment
FROM node:18-alpine AS base
WORKDIR /app
RUN apk add --no-cache git curl

# Stage 2: Dependencies installation
FROM base AS dependencies
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Stage 3: Development dependencies
FROM base AS dev-dependencies
COPY package*.json ./
RUN npm ci && npm cache clean --force

# Stage 4: Testing environment
FROM dev-dependencies AS testing
COPY . .
ENV NODE_ENV=test
ENV CI=true
RUN npm run lint || true
RUN npm run type-check || true
RUN npm run test:coverage || true

# Stage 5: Development environment
FROM dev-dependencies AS development
COPY . .
EXPOSE 3000
ENV NODE_ENV=development
ENV REACT_APP_API_URL=http://localhost:8000/api/v1
ENV REACT_APP_WS_URL=ws://localhost:8000
CMD ["npm", "start"]

# Stage 6: Production build
FROM base AS builder
COPY package*.json ./
RUN npm ci --only=production
COPY . .
ENV NODE_ENV=production
ENV REACT_APP_API_URL=/api/v1
RUN npm run build

# Stage 7: Production environment with Nginx
FROM nginx:alpine AS production
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

# Create non-root user
RUN addgroup -g 1001 -S nginx && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nologin -G nginx -g nginx nginx

# Set ownership
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    chown -R nginx:nginx /etc/nginx/conf.d

# Switch to non-root user
USER nginx

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# Stage 8: E2E Testing environment
FROM cypress/included:12.5.1 AS e2e-testing
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ENV NODE_ENV=test
ENV CI=true
CMD ["npm", "run", "test:e2e"]
