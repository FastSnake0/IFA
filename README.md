# IFA

Cистема генерации изображений на основе модели Stable Diffusion. 
Поддерживает IP Adapter, внедрение в cross-attention (KV injection), работу с пользователями и базой изображений.

## Архитектура

Gateway (FastAPI)
↓
Backend (FastAPI + SQLAlchemy + DB)
↓
Model (FastAPI + Diffusers + IP Adapter)

## Эндпоинты

### POST /generate

- **Описание**: Генерация изображения с заданными параметрами.
- **Запрос**:

  ```json
  {
    "username": "alice",
    "model": "runwayml/stable-diffusion-v1-5",
    "pos_prompt": "a cat sitting on a couch",
    "ng_prompt": "blurry",
    "ip": null,
    "inject": null
  }
  ```

### GET /models

- **Описание**: Получение списка доступных моделей генераци.
- **Запрос**:

  ```json
  {
  "models": [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0"
  ]
  }
  ```

### GET /users

- **Описание**: Получение списка всех пользователей.
- **Запрос**:

  ```json
  [
  {"username": "alice", "created_at": "..."},
  {"username": "bob",   "created_at": "..."}
  ]
  ```

### GET /users/{username}

- **Описание**: Статистика конкретного пользователя.
- **Запрос**:

  ```json
  {
  "username": "alice",
  "created_at": "2025-06-23T12:34:56Z",
  "images_generated": 42
  }
  ```

### GET /images

- **Описание**: Просмотр изображений с фильтрацией.
- **Параметры**:
    - username (опционально)
    - date (YYYY‑MM‑DD, опционально)
- **Запрос**:

  ```json
  {
  "username": "alice",
  "created_at": "2025-06-23T12:34:56Z",
  "images_generated": 42
  }
  ```

### GET /images/{image_id}

- **Описание**: Метаданные конкретного изображения.
- **Запрос**:

  ```json
  {
  "id": 1,
  "username": "alice",
  "model": "…",
  "prompt": "…",
  "created_at": "…"
  }
  ```

## Пример запроса

```bash
curl -X POST http://localhost:8000/generate \
-H "Content-Type: application/json" \
-d '{
  "username": "alice",
  "model": "runwayml/stable-diffusion-v1-5",
  "pos_prompt": "a cat sitting on a couch",
  "ng_prompt": "blurry",
  "ip": null,
  "inject": null
}' --output result.jpg
```

## Поля запроса

| Поле       | Тип        | Обязательное | Описание |
|------------|------------|--------------|----------|
| `username` | `string`   | ✅           | Имя пользователя. Если нового пользователя нет — создаётся. |
| `model`    | `string`   | ✅           | Название модели (например, `runwayml/stable-diffusion-v1-5` или `stabilityai/stable-diffusion-xl-base-1.0`). |
| `pos_prompt` | `string` | ✅           | Основной текстовый запрос (positive prompt). |
| `ng_prompt`  | `string` | ❌           | Негативный prompt, чтобы исключить нежелательные черты. |
| `ip`       | `object` или `null` | ❌ | Настройка IP Adapter: `{ "image": <base64-строка>, "scale": float }`. |
| `inject`   | `object` или `null` | ❌ | Настройка для KV-инъекции: `{ "prompt": string, "scale": float, "index": int }`. |

## Сборка

```bash
docker-compose up --build
```

## research

Наработки по исследованиям.
