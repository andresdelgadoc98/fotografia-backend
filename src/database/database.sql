CREATE TABLE IF NOT EXISTS image_analysis (
    image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    description TEXT,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    tags JSON,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS folders (
    id SERIAL PRIMARY KEY,
    root_path TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP NOT NULL,
    status VARCHAR(20) DEFAULT 'idle'
);

CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    folder_id INTEGER REFERENCES folders(id),
    abs_path TEXT UNIQUE NOT NULL,
    size_bytes BIGINT,
    mtime BIGINT,
    sha256 TEXT,
    status TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);
