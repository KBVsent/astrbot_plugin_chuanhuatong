# 传话筒（astrbot_plugin_chuanhuatong）

> 当前版本：**v1.3.0**

将 AstrBot 的所有纯文本回复转换成带立绘的 GalGame 风聊天框图片，支持情绪差分、多段富文本、拖拽式 WebUI 布局设置与自定义模板组件。

## 功能特性

- ✅ 拦截 Bot 输出的文本，在发送前统一转为图片，失败时自动降级为原始文本
- 🎨 自动抽取背景 / 立绘素材：`background/`、`renwulihui/<情绪>/`，WebUI 可实时预览
- 😳 支持 LLM 情绪提示：在回答中嵌入 `&happy&`、`&shy&` 等标签即可切换立绘
- 🔧 内置 WebUI，可拖拽主文本框、立绘，以及任意数量的文本 / 组件 / 毛玻璃图层，直接写回配置
- 📝 文本层支持自定义字体、颜色、**描边宽度 / 颜色**，可通过拖拽边框控制自动换行
- 🧊 毛玻璃效果独立成图层，可与主文本框分离自由摆放
- 🍮 完全基于本地 Pillow 渲染，无需 Playwright；如渲染失败会自动退回纯文本
- 🧰 端口/token、情绪映射、渲染字符阈值等在 `_conf_schema.json` 中调节，其余布局交给 WebUI

## 安装

1. 将整个目录放入 `AstrBot/data/plugins/astrbot_plugin_chuanhuatong`
2. 在 AstrBot WebUI → 插件管理中启用插件，并根据需要修改配置
3. 如果要启用 WebUI，请确保 `webui_host`/`webui_port` 未被占用

## WebUI

- 默认监听 `http://127.0.0.1:18765`，可通过 `?token=xxx` 或 `Authorization: Bearer xxx` 访问
- 画布内可拖拽 / 缩放：主文本框、立绘框及所有文本 / 组件 / 毛玻璃层
- 图层列表支持查看/选择/删除，类似 Photoshop，可调节 z-index、透明度、可见性
- 表单中可指定字体（系统字体 / 上传字体）、背景资源等；**新增文本层支持描边 / 自动换行预览**
- “上传资产”区可直接上传 PNG/WebP/GIF 到组件目录，或上传 TTF/TTC/OTF 字体到字体目录
- 配置保存在 `data/plugin_data/astrbot_plugin_chuanhuatong/layout_state.json`，刷新 / 重载插件不会丢失

## 情绪标签

- 插件会在 LLM 请求阶段追加提示，要求模型在回答中插入 `&tag&`
- 默认标签：`&happy&`、`&sad&`、`&shy&`、`&surprise&`、`&angry&`、`&neutral&`
- 也可以关闭自动提示，然后在其他插件/Prompt 中手写标签

## 资源放置

```text
astrbot_plugin_chuanhuatong/
├── background/          # 背景图，支持 png/jpg/webp，随机抽取
├── renwulihui/          # 立绘根目录，按情绪或自定义子文件夹分类
└── zujian/              # 内置组件（模板框、装饰按钮等 PNG/WebP/GIF）
```

- 立绘建议使用透明 PNG，文件名任意。可以自由新增子目录，然后在配置的 `emotion_sets` 中指向这些目录。  
- **内置模板图片（示例）：**`名称框.png`、`底框.png`、`线索.png`、`设置.png` 等，请直接放在插件目录下的 `zujian/` 中，  
  即：`AstrBot/data/plugins/astrbot_plugin_chuanhuatong/zujian/名称框.png`。  
  这样就能和默认布局中预置的组件图层一一对应，第一次打开 WebUI 就能看到完整模板。

### 数据目录（自动创建）

```text
AstBot/data/plugin_data/astrbot_plugin_chuanhuatong/
├── layout_state.json   # WebUI 保存的布局
├── zujian/             # WebUI 上传的额外组件（PNG/WebP/GIF）
└── fonts/              # WebUI 上传的字体文件
```

将素材 / 字体直接放入上述目录也可以被 WebUI 识别；用户上传的组件与字体会与插件内置的文件一起出现在下拉列表中。

## 配置重点

- `enable_render`：是否拦截文本并尝试渲染为图片
- `render_char_threshold`：**渲染字符阈值**，0 为不限制，超过该长度则直接发送纯文本（默认约 60 个汉字）
- `enable_emotion_prompt`：是否自动注入情绪提示语（配置项 `emotion_prompt_template` 支持 `{tags}` 占位符）
- `emotion_sets`：一个列表，用来声明可用的情绪标签 / 对应立绘文件夹 / 角标颜色，可根据需求增删
- `font_path`：可选。若需要在本地 Pillow 备用渲染里使用自定义字体，填写字体文件的绝对路径
- `webui_port` / `webui_token`：自定义 WebUI 端口和访问口令；布局细节（文本框、额外文本、毛玻璃等）请在 WebUI 中拖拽完成

## 注意事项

- 插件仅依赖 Pillow；如需使用自定义字体，可在 WebUI 中上传或设置 `font_path`
- 如果未检测到背景或立绘文件，会自动降级为纯背景/无立绘模式
- 若在其他插件中也修改了 `event.get_result()`，请注意执行顺序及 `event.stop_event()` 的影响
