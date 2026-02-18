-- Metal DiskANN Benchmark
-- Benchmarks DiskANN search at different dimensions.
-- The optimizer rewrites ORDER BY array_distance LIMIT k → ANN index scan.

.timer on

-- ========================================
-- Setup: 10000 vectors, dim=128
-- ========================================
SELECT '=== dim=128, 10000 vectors ===' as status;
SELECT setseed(0.42);

CREATE TABLE v128 AS
SELECT i AS id,
  list_transform(range(128), x -> random()::FLOAT)::FLOAT[128] AS embedding
FROM range(0, 10000) t(i);

CREATE INDEX v128_idx ON v128 USING DISKANN (embedding)
  WITH (max_degree = 64, build_complexity = 100);

-- Warm up
SELECT count(*) FROM (
  SELECT id FROM v128
  ORDER BY array_distance(embedding, (SELECT embedding FROM v128 WHERE id=0))
  LIMIT 10
);

-- Benchmark: 50 sequential queries via optimizer
SELECT 'dim=128: 50 queries x k=10' as status;
CREATE TABLE r128 AS
SELECT q.id as qid, v.id as vid, array_distance(v.embedding, q.embedding) as dist
FROM (SELECT * FROM v128 WHERE id < 50) q,
LATERAL (
  SELECT id, embedding FROM v128
  ORDER BY array_distance(embedding, q.embedding)
  LIMIT 10
) v;
SELECT count(*) as results FROM r128;

DROP TABLE r128;
DROP INDEX v128_idx;
DROP TABLE v128;

-- ========================================
-- Setup: 10000 vectors, dim=256
-- ========================================
SELECT '=== dim=256, 10000 vectors ===' as status;
SELECT setseed(0.42);

CREATE TABLE v256 AS
SELECT i AS id,
  list_transform(range(256), x -> random()::FLOAT)::FLOAT[256] AS embedding
FROM range(0, 10000) t(i);

CREATE INDEX v256_idx ON v256 USING DISKANN (embedding)
  WITH (max_degree = 64, build_complexity = 100);

SELECT 'dim=256: 50 queries x k=10' as status;
CREATE TABLE r256 AS
SELECT q.id as qid, v.id as vid, array_distance(v.embedding, q.embedding) as dist
FROM (SELECT * FROM v256 WHERE id < 50) q,
LATERAL (
  SELECT id, embedding FROM v256
  ORDER BY array_distance(embedding, q.embedding)
  LIMIT 10
) v;
SELECT count(*) as results FROM r256;

DROP TABLE r256;
DROP INDEX v256_idx;
DROP TABLE v256;

-- ========================================
-- Setup: 10000 vectors, dim=768
-- ========================================
SELECT '=== dim=768, 10000 vectors ===' as status;
SELECT setseed(0.42);

CREATE TABLE v768 AS
SELECT i AS id,
  list_transform(range(768), x -> random()::FLOAT)::FLOAT[768] AS embedding
FROM range(0, 10000) t(i);

CREATE INDEX v768_idx ON v768 USING DISKANN (embedding)
  WITH (max_degree = 64, build_complexity = 100);

SELECT 'dim=768: 50 queries x k=10' as status;
CREATE TABLE r768 AS
SELECT q.id as qid, v.id as vid, array_distance(v.embedding, q.embedding) as dist
FROM (SELECT * FROM v768 WHERE id < 50) q,
LATERAL (
  SELECT id, embedding FROM v768
  ORDER BY array_distance(embedding, q.embedding)
  LIMIT 10
) v;
SELECT count(*) as results FROM r768;

DROP TABLE r768;
DROP INDEX v768_idx;
DROP TABLE v768;

SELECT 'Done' as status;
