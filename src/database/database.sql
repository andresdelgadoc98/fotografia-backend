CREATE TABLE IF NOT EXISTS image_analysis (
    image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    description TEXT,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    tags JSON,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
