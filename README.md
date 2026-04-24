# 机器人状态曲线查看器

一个面向 Ubuntu / Windows 的桌面 GUI，用于读取当前目录中这类“XML 头 + 时序数据体”的机器人日志 `txt` 文件，并交互式查看多条运行状态曲线。

## 功能

- 打开单个日志文件
- 自动解析 XML 信号定义与 `ID` 列映射
- 左侧字段树筛选信号，右侧叠加显示多条曲线
- 支持搜索、缩放、平移、重置视图
- 显示相对起始时间轴，并在状态栏展示原始起止时间
- 对少量坏行进行跳过统计，不影响整体数据查看

## 安装

```bash
python3 -m pip install -r requirements.txt
```

Windows 下使用：

```powershell
py -m pip install -r requirements.txt
```

## 运行

不带文件启动：

```bash
python3 -m app
```

直接打开文件：

```bash
python3 -m app data_1min_4ms_20260228132326.txt
```

Windows 下使用：

```powershell
py -m app .\data_1min_4ms_20260228132326.txt
```

## 数据格式

当前版本仅支持如下结构：

1. 文件前半段为 XML 信号定义
2. 中间使用一行纯 `*` 作为分隔
3. 下一行是 `ID` 表头
4. 后续每行是一个采样点
5. 第一列为时间戳，格式为 `YYYYMMDDHHMMSSmmm`

## 测试

解析层测试：

```bash
python3 -m unittest discover -s tests
```

GUI 依赖未安装时，解析层测试仍可单独运行；桌面界面需要先安装 `PySide6` 和 `pyqtgraph`。
