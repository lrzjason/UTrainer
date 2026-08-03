-- ScheduledTrainer orchestrator schema (P1)
-- 七表：projects / tasks / hooks / task_config_kv / prompts / heartbeats / gpu_snapshots
-- 来源：doc/Improvement_k3.md §3.2

-- 项目：任务的上层组织单位
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',   -- active / archived
    default_model TEXT,                            -- 项目级默认值，任务可覆盖
    tags        TEXT DEFAULT '[]',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 任务定义：JSON config 导入后物化为一行，必属于某个 project
CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id),
    name          TEXT NOT NULL,
    model         TEXT NOT NULL,            -- flux2_klein / qwen_image / ...
    config_json   TEXT NOT NULL,            -- 完整训练配置快照
    status        TEXT NOT NULL DEFAULT 'pending',
                  -- pending / scheduled / running / suspended / paused /
                  -- done / failed / cancelled / waiting_gpu
    priority      INTEGER NOT NULL DEFAULT 100,   -- 越小越优先
    depends_on    INTEGER REFERENCES tasks(id),   -- 串行链：前驱任务
    resume_from   TEXT,                     -- 权重路径（可引用前驱产物）
    resume_mode   TEXT DEFAULT 'weights',   -- weights / full（是否带优化器）
    cron          TEXT,                     -- 定时表达式，NULL = 立即/依赖触发
    at            TEXT,                     -- 一次性触发 ISO 时间（P3），与 cron 二选一
    allow_parallel INTEGER NOT NULL DEFAULT 0, -- 定时任务是否允许与在跑任务并行
    gpus          INTEGER NOT NULL DEFAULT 1,  -- 本任务需要几张卡（P5），默认 1=单卡
    gpu_ids       TEXT,                     -- 钉卡 CSV "0,1"（可空，NULL=自动选卡）
    wandb_run_name TEXT,
    restart_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    started_at    TEXT, finished_at TEXT,
    source_file   TEXT,                     -- 来自 inbox/ 的原始文件
    error         TEXT
);

-- hook 指令队列：orchestrator/CLI/前端 写入，worker 每 step 消费
CREATE TABLE IF NOT EXISTS hooks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    INTEGER NOT NULL REFERENCES tasks(id),
    type       TEXT NOT NULL,
               -- sample / sample_from_weights / save / restore /
               -- patch_config / suspend / stop
    payload    TEXT NOT NULL DEFAULT '{}',  -- JSON
    status     TEXT NOT NULL DEFAULT 'queued',  -- queued / acked / done / failed
    result     TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    acked_at   TEXT, done_at TEXT
);

-- 动态配置：worker 运行期实际生效的 config 以这里为准
CREATE TABLE IF NOT EXISTS task_config_kv (
    task_id  INTEGER NOT NULL REFERENCES tasks(id),
    key      TEXT NOT NULL,                 -- 点路径，如 training.learning_rate
    value    TEXT NOT NULL,                 -- JSON 编码
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (task_id, key)
);

-- 采样提示词缓存（项目级共享）
CREATE TABLE IF NOT EXISTS prompts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id),
    tag        TEXT,                            -- 分组标签
    text       TEXT NOT NULL,
    negative   TEXT,
    meta       TEXT DEFAULT '{}'                -- 分辨率、步数、seed 等
);

-- 训练进度心跳（worker 定期上报，orchestrator 判活 + 前端画曲线）
CREATE TABLE IF NOT EXISTS heartbeats (
    task_id  INTEGER NOT NULL,
    step     INTEGER NOT NULL,
    loss     REAL,
    lr       REAL,
    vram_mb  INTEGER,
    ts       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- GPU 资源快照（orchestrator 周期采集，前端展示 + GPU Guard 决策依据）
CREATE TABLE IF NOT EXISTS gpu_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    gpu_index   INTEGER NOT NULL DEFAULT 0,
    total_mb    INTEGER NOT NULL,
    used_mb     INTEGER NOT NULL,
    free_mb     INTEGER NOT NULL,
    util_pct    INTEGER,
    ts          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 常用查询索引
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_depends ON tasks(depends_on);
CREATE INDEX IF NOT EXISTS idx_hooks_task_status ON hooks(task_id, status);
CREATE INDEX IF NOT EXISTS idx_heartbeats_task_ts ON heartbeats(task_id, ts);
