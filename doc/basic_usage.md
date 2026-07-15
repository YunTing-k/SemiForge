## 基本使用 | Basic Usage

### 命令行参数 | CLI Arguments

```bash
python -m src.main                                # 启动 Agent / Launch agent
python -m src.main -l                             # 启用开发者日志 / Enable developer logs
python -m src.main -r <UUID>                      # 恢复指定会话 / Resume a session
python -m src.main --nosystem                     # 禁用系统提示词 / Disable system prompts
python -m src.main --notools                      # 禁用工具 / Disable agent tools
python -m src.main --nocrons                      # 禁用定时任务 / Disable cron tasks
python -m src.main --noskills                     # 禁用技能 / Disable skills
python -m src.main --nomcps                       # 禁用 MCP / Disable MCPs
python -m src.main --dangerously_allow_all        # ⚠️ 允许所有权限 / Allow all permissions
```

### 子命令 | Sub-commands

```bash
# 会话管理 | Session management
python -m src.main session list                   # 列出所有会话 / List all sessions
python -m src.main session remove <UUID>          # 删除指定会话 / Remove a session

# 持久化定时任务管理 | Durable cron management
python -m src.main cron list                      # 列出持久化定时任务 / List durable cron tasks
python -m src.main cron remove <ID>               # 删除持久化定时任务 / Remove a durable cron task

# 技能管理 | Skill management
python -m src.main skill list                     # 列出所有可用技能 / List all available skills

# MCP 管理 | MCP management
python -m src.main mcp list                       # 列出所有 MCP 服务器 / List all MCP servers
python -m src.main mcp add <name> <type> <params> # 添加 MCP 服务器 / Add an MCP server
python -m src.main mcp toggle <name>              # 启用/禁用 MCP 服务器 / Enable/disable an MCP server
python -m src.main mcp remove <name>              # 移除 MCP 服务器 / Remove an MCP server
```

### 内建命令 | Built-in Commands

在 Agent 交互界面中，所有命令以 `/` 开头：
All commands start with `/` in the agent interaction interface:

| 命令 Command | 功能 Description |
|---------|-------------|
| `/help` | 查看所有可用命令 / Show all available commands |
| `/context` | 查看 Token 用量与上下文统计 / View token usage and context stats |
| `/fread_list` | 查看所有已读文件 / View all read files |
| `/url_caches` | 查看缓存的 URL / View cached URLs |
| `/session_list` | 查看所有会话 / View all sessions |
| `/session_remove <UUID>` | 删除指定会话 / Remove a session |
| `/readonly_list` | 查看只读路径 / View read-only paths |
| `/readonly_add <PATH> [PATH...]` | 添加只读路径 / Add read-only paths |
| `/readonly_remove <idx> [idx...]` | 移除只读路径 / Remove read-only paths |
| `/permission_list` | 查看权限配置 / View permission configs |
| `/permission_toggle <NAME>` | 切换权限开关 / Toggle a permission |
| `/skill_list` | 列出可用技能 / List available skills |
| `/skills_loaded` | 列出已加载技能 / List loaded skills |
| `/<skill_name>` | 手动加载技能 / Manually load a skill |
| `/mcp_list` | 查看 MCP 信息 / View MCP information |
| `/cron_list` | 查看定时任务列表 / View cron task list |
| `/cron_remove <ID>` | 删除定时任务 / Remove a cron task |
| `/task_list` | 查看未归档的 Agent 任务 / List non-archived agent tasks |
| `/task_list_all` | 查看所有历史 Agent 任务 / List all history agent tasks |
| `/agent_list` | 查看所有子Agent（活跃和已归档）/ List all subagents (active and archived) |
| `/update_title` | 用 LLM 自动更新当前会话标题 / Auto-update current session title with LLM |
| `/set_title <TITLE>` | 手动设置当前会话标题 / Manually set current session title |
| `/dg_tft_list` | 列出所有 DG-TFT 工作流历史运行 / List all DG-TFT workflow runs |
| `/dg_tft_status <run_id\|prep_id>` | 查询 DG-TFT 工作流运行状态 / Query a DG-TFT workflow run status |
| `/dg_tft_report <run_id>` | 查看 DG-TFT 工作流运行报告 / Read a DG-TFT workflow report |
