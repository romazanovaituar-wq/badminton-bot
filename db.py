"""
RallyIQ — слой работы с базой данных PostgreSQL.
Хранит пользователей, их кредиты, язык и историю анализов.
"""
import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Пул соединений — переиспользуем подключения вместо открытия нового на каждый запрос
_pool: SimpleConnectionPool | None = None


def init_pool() -> None:
    """Создаёт пул соединений. Вызывается один раз при старте."""
    global _pool
    if not DATABASE_URL:
        logger.warning("DATABASE_URL не задан — база данных недоступна")
        return
    _pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)
    logger.info("Пул соединений с PostgreSQL создан")


@contextmanager
def get_conn():
    """Безопасно берёт соединение из пула и возвращает обратно."""
    if _pool is None:
        raise RuntimeError("Пул соединений не инициализирован")
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def init_schema() -> None:
    """Создаёт таблицы если их ещё нет."""
    if _pool is None:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id      BIGINT PRIMARY KEY,
                    username     TEXT,
                    first_name   TEXT,
                    lang         TEXT DEFAULT 'ru',
                    credits      INTEGER DEFAULT 0,
                    free_used    BOOLEAN DEFAULT FALSE,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id           SERIAL PRIMARY KEY,
                    user_id      BIGINT REFERENCES users(user_id),
                    frames       INTEGER,
                    shirt        TEXT,
                    status       TEXT DEFAULT 'done',
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id           SERIAL PRIMARY KEY,
                    user_id      BIGINT REFERENCES users(user_id),
                    amount       INTEGER,
                    credits      INTEGER,
                    provider     TEXT,
                    external_id  TEXT UNIQUE,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)
    logger.info("Схема базы данных проверена/создана")


def ensure_user(user_id: int, username: str | None, first_name: str | None) -> None:
    """Создаёт пользователя если его нет."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (user_id, username, first_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET username = EXCLUDED.username,
                    first_name = EXCLUDED.first_name;
            """, (user_id, username, first_name))


def get_user(user_id: int) -> dict | None:
    """Возвращает данные пользователя."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s;", (user_id,))
            return cur.fetchone()


def get_lang(user_id: int) -> str:
    user = get_user(user_id)
    return user["lang"] if user else "ru"


def set_lang(user_id: int, lang: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET lang = %s WHERE user_id = %s;", (lang, user_id))


def get_credits(user_id: int) -> int:
    user = get_user(user_id)
    return user["credits"] if user else 0


def add_credits(user_id: int, amount: int) -> int:
    """Добавляет кредиты и возвращает новый баланс."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET credits = credits + %s
                WHERE user_id = %s RETURNING credits;
            """, (amount, user_id))
            row = cur.fetchone()
            return row[0] if row else 0


def consume_credit(user_id: int) -> bool:
    """
    Атомарно списывает 1 кредит. Возвращает True если успешно,
    False если кредитов не было. Защищено от гонки запросов.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET credits = credits - 1
                WHERE user_id = %s AND credits > 0
                RETURNING credits;
            """, (user_id,))
            row = cur.fetchone()
            return row is not None


def grant_free(user_id: int) -> bool:
    """
    Выдаёт 1 бесплатный анализ если ещё не выдавался.
    Возвращает True если выдан, False если уже был использован.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET credits = credits + 1, free_used = TRUE
                WHERE user_id = %s AND free_used = FALSE
                RETURNING credits;
            """, (user_id,))
            row = cur.fetchone()
            return row is not None


def log_analysis(user_id: int, frames: int, shirt: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analyses (user_id, frames, shirt)
                VALUES (%s, %s, %s);
            """, (user_id, frames, shirt))


# ==================================================
# АДМИН-ФУНКЦИИ
# ==================================================
def get_stats() -> dict:
    """Общая статистика для админа."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS total_users FROM users;")
            total_users = cur.fetchone()["total_users"]

            cur.execute("SELECT COUNT(*) AS total_analyses FROM analyses;")
            total_analyses = cur.fetchone()["total_analyses"]

            cur.execute("SELECT COALESCE(SUM(credits), 0) AS total_credits FROM users;")
            total_credits = cur.fetchone()["total_credits"]

            cur.execute("""
                SELECT COUNT(*) AS active FROM analyses
                WHERE created_at > NOW() - INTERVAL '7 days';
            """)
            active_week = cur.fetchone()["active"]

            return {
                "total_users": total_users,
                "total_analyses": total_analyses,
                "total_credits": total_credits,
                "active_week": active_week,
            }


def get_recent_users(limit: int = 15) -> list[dict]:
    """Последние зарегистрированные пользователи."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT u.user_id, u.username, u.first_name, u.lang, u.credits,
                       COUNT(a.id) AS analyses_count
                FROM users u
                LEFT JOIN analyses a ON a.user_id = u.user_id
                GROUP BY u.user_id
                ORDER BY u.created_at DESC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()


def find_user_by_username(username: str) -> dict | None:
    """Находит пользователя по username (без @)."""
    username = username.lstrip("@")
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE username = %s;", (username,))
            return cur.fetchone()


def add_credits_by_id(user_id: int, amount: int) -> int | None:
    """Начисляет кредиты по user_id. Возвращает новый баланс или None если юзера нет."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET credits = credits + %s
                WHERE user_id = %s RETURNING credits;
            """, (amount, user_id))
            row = cur.fetchone()
            return row[0] if row else None
