"""Start this UI with the project's Python and ``-m streamlit run app.py``."""
from __future__ import annotations

import io
import threading
from pathlib import Path

import streamlit as st
from fashion_demo.pipeline import FashionPipeline, ROOT, png_bytes, read_image

st.set_page_config(page_title="时尚图像检测与分割", page_icon="👗", layout="wide")


@st.cache_resource
def get_pipeline():
    return FashionPipeline(), threading.Lock()


st.title("时尚图像检测与分割")
st.write("上传一张穿搭图片，查看人物与配饰检测、人物区域分割及叠加效果。")
st.caption("课程基础版 · Faster R-CNN + DeepLabV3 · 分割结果为整个人物区域，不区分上衣、裤子或裙子。")
with st.sidebar:
    st.header("图片与参数")
    upload = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "webp"])
    examples = sorted((ROOT / "examples").glob("*.jpg"))
    example_name = st.selectbox("或选择示例", ["请选择"] + [p.name for p in examples])
    threshold = st.slider("检测置信度阈值", 0.05, 0.95, 0.50, 0.05)
    alpha = st.slider("分割掩码透明度", 0.0, 1.0, 0.45, 0.05)
    st.caption("首次运行需要下载模型。下载完成后可使用本地缓存离线演示。")

source = io.BytesIO(upload.getvalue()) if upload is not None else ROOT / "examples" / example_name if example_name != "请选择" else None
if source is not None:
    try:
        original = read_image(source)
        with st.expander("查看输入图片", expanded="result" not in st.session_state):
            st.image(original, caption="当前输入图片", width=360)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
else:
    st.info("请上传图片或选择左侧示例，然后点击“开始分析”。")

if st.button("开始分析", type="primary", disabled=source is None):
    st.session_state.pop("result", None)
    try:
        with st.spinner("正在加载模型并分析图片，CPU推理可能需要一些时间……"):
            pipeline, inference_lock = get_pipeline()
            with inference_lock:
                st.session_state.result = pipeline.run(original, threshold, alpha)
            st.session_state.result_source = upload.name if upload else example_name
    except Exception as exc:
        st.error(f"分析失败：{exc}")
        st.info("若模型下载失败，请检查网络后重试；若内存不足，请缩小图片。也可以查看README中的排查步骤。")

if "result" in st.session_state:
    result = st.session_state.result
    st.subheader(f"分析结果 · {st.session_state.result_source}")
    st.caption(f"本次参数：置信度 {result.metadata['threshold']:.2f}，透明度 {result.metadata['alpha']:.2f}。修改输入或参数后请重新点击分析。")
    a, b, c = st.columns(3)
    a.metric("检测目标", len(result.metadata["detections"]))
    b.metric("推理耗时", f"{result.metadata['inference_seconds']:.2f} 秒")
    c.metric("运行设备", result.metadata["device"].upper())
    tabs = st.tabs(["叠加效果", "目标检测", "人物区域分割", "原图"])
    for tab, name in zip(tabs, ("combined", "detection", "segmentation", "original")):
        with tab:
            st.image(getattr(result, name), width="stretch")
    if result.metadata["detections"]:
        st.dataframe([{"类别": d["category_zh"], "置信度": d["score"]} for d in result.metadata["detections"]], hide_index=True, width="stretch")
    else:
        st.info("当前阈值下未检测到指定类别。可以降低阈值或更换图片。")
    if result.metadata["person_area_ratio"] == 0:
        st.info("分割模型未识别出人物区域，本次分割图保留原图。")
    st.caption("耗时包含预处理，不包含模型加载与结果绘制；置信度不等同于准确率。")
    st.download_button("下载叠加结果 PNG", png_bytes(result.combined), "combined.png", "image/png")
    st.download_button("下载全部结果 ZIP", result.archive(), "fashion_results.zip", "application/zip")
