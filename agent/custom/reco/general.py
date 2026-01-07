import re
import sys
import json
import random
from typing import Any, Dict, List, Union, Optional

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.define import RectType
from utils.logger import logger

@AgentServer.custom_recognition("ColorOCR")
class ColorOCR(CustomRecognition):
    """
    颜色过滤后进行OCR识别。

    参数格式:
    {
        "target_color": [R, G, B],
        "tolerance": int,
        "recognition": string
    }

    字段说明:
    - target_color: 目标颜色RGB值，默认 [255, 255, 255] (白色)
    - tolerance: 颜色容差，默认55
    - recognition: 要运行的OCR识别节点名称
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        try:
            params = json.loads(argv.custom_recognition_param)

            # 获取参数，默认过滤白色
            target_color = params.get("target_color", [255, 255, 255])
            tolerance = params.get("tolerance", 55)
            recognition_node = params.get("recognition")

            if not target_color or len(target_color) != 3:
                logger.error(f"无效的target_color参数: {target_color}")
                return None

            if not recognition_node:
                logger.error("未提供recognition参数")
                return None

            # 获取图像
            img = argv.image

            # 定义目标颜色和颜色容差
            target_color_array = np.array(target_color)

            # 创建颜色过滤掩码
            lower_bound = np.maximum(target_color_array - tolerance, 0)
            upper_bound = np.minimum(target_color_array + tolerance, 255)

            # 创建掩码：保留在目标颜色范围内的像素
            color_mask = np.all((img >= lower_bound) & (img <= upper_bound), axis=-1)

            # 处理图像：目标颜色变成黑色，其他颜色变成白色
            # 创建一个全白图像
            processed_img = np.full_like(img, 255, dtype=np.uint8)
            # 将匹配目标颜色的像素设置为黑色
            processed_img[color_mask] = 0

            # 在处理后的图像上运行OCR识别
            reco_detail = context.run_recognition(recognition_node, processed_img)

            if reco_detail and reco_detail.hit:
                logger.debug(f"ColorOCR识别成功: {reco_detail.best_result.text}")
                return CustomRecognition.AnalyzeResult(
                    box=reco_detail.box, detail=reco_detail.raw_detail
                )
            else:
                return None

        except Exception as e:
            logger.error(f"ColorOCR识别失败: {e}")
            return None


@AgentServer.custom_recognition("ColorOCRWithFallback")
class ColorOCRWithFallback(CustomRecognition):
    """
    颜色过滤OCR识别，失败后自动fallback到纯OCR识别。

    参数格式:
    {
        "target_color": [R, G, B],
        "tolerance": int,
        "recognition": string
    }

    字段说明:
    - target_color: 目标颜色RGB值，默认 [255, 255, 255] (白色)
    - tolerance: 颜色容差，默认55
    - recognition: 要运行的OCR识别节点名称

    工作流程:
    1. 先尝试颜色过滤后的OCR识别（ColorOCR）
    2. 如果失败，再尝试纯OCR识别（不过滤颜色）
    3. 返回任意一种成功的结果
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        try:
            params = json.loads(argv.custom_recognition_param)

            # 获取参数
            target_color = params.get("target_color", [255, 255, 255])
            tolerance = params.get("tolerance", 55)
            recognition_node = params.get("recognition")

            if not target_color or len(target_color) != 3:
                logger.error(f"无效的target_color参数: {target_color}")
                return None

            if not recognition_node:
                logger.error("未提供recognition参数")
                return None

            # 获取图像
            img = argv.image

            # 第一步：尝试 ColorOCR（颜色过滤）
            target_color_array = np.array(target_color)
            lower_bound = np.maximum(target_color_array - tolerance, 0)
            upper_bound = np.minimum(target_color_array + tolerance, 255)

            # 创建掩码：保留在目标颜色范围内的像素
            color_mask = np.all((img >= lower_bound) & (img <= upper_bound), axis=-1)

            # 处理图像：目标颜色变成黑色，其他颜色变成白色
            processed_img = np.full_like(img, 255, dtype=np.uint8)
            processed_img[color_mask] = 0

            # 在处理后的图像上运行OCR识别
            reco_detail = context.run_recognition(recognition_node, processed_img)

            if reco_detail and reco_detail.hit:
                logger.debug(f"ColorOCRWithFallback: ColorOCR识别成功")
                return CustomRecognition.AnalyzeResult(
                    box=reco_detail.box,
                    detail={
                        "method": "color_ocr",
                        "raw_detail": reco_detail.raw_detail,
                    },
                )

            # 第二步：ColorOCR失败，尝试纯OCR（不过滤颜色）
            reco_detail = context.run_recognition(
                recognition_node,
                img,
                {
                    "TargetStageName_OCR": {
                        "recognition": {"param": {"roi": [63, 533, 1156, 62]}}
                    }
                },
            )

            if reco_detail and reco_detail.hit:
                logger.debug(f"ColorOCRWithFallback: 纯OCR识别成功")
                return CustomRecognition.AnalyzeResult(
                    box=reco_detail.box,
                    detail={
                        "method": "pure_ocr",
                        "raw_detail": reco_detail.raw_detail,
                    },
                )
            else:
                return None

        except Exception as e:
            logger.error(f"ColorOCRWithFallback识别失败: {e}")
            return None


@AgentServer.custom_recognition("RecognitionResultsArray")
class RecognitionResultsArray(CustomRecognition):
    """
    遍历识别节点的所有结果并以数组形式输出。

    参数格式:
    {
        "recognition": string,
        "index": int
    }

    字段说明:
    - recognition: 要运行的识别节点名称
    - index: 对于And节点，指定从哪个子识别中获取结果（默认0）

    返回格式:
    {
        "results": [
            {
                "text": string,  // 识别文本
                "box": [x, y, w, h],  // 识别框
                "score": float  // 置信度分数
            },
            ...
        ]
    }
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        try:
            params = json.loads(argv.custom_recognition_param)

            # 获取参数
            recognition_node = params.get("recognition")
            index = params.get("index", 0)

            if not recognition_node:
                logger.error("未提供recognition参数")
                return None

            # 获取图像
            img = argv.image

            # 运行识别节点
            reco_detail = context.run_recognition(recognition_node, img)

            if reco_detail and reco_detail.hit:
                # 提取所有识别结果
                results = []
                
                # 检查raw_detail是否包含多个结果
                if isinstance(reco_detail.raw_detail, dict):
                    # 对于And节点，使用指定的index获取对应子识别的结果
                    if "sub_details" in reco_detail.raw_detail and isinstance(reco_detail.raw_detail["sub_details"], list):
                        sub_details = reco_detail.raw_detail["sub_details"]
                        if 0 <= index < len(sub_details):
                            sub_detail = sub_details[index]
                            if isinstance(sub_detail, dict):
                                # 处理子识别的结果
                                text_results = sub_detail.get("text_results", [])
                                if isinstance(text_results, list):
                                    for result in text_results:
                                        if isinstance(result, dict):
                                            text = result.get("text", "")
                                            box = result.get("box", [])
                                            score = result.get("score", 0.0)
                                            results.append({
                                                "text": text,
                                                "box": box,
                                                "score": score
                                            })
                                # 如果子识别没有text_results字段，尝试获取其其他结果
                                elif not results:
                                    text = sub_detail.get("text", "")
                                    box = sub_detail.get("box", [])
                                    score = sub_detail.get("score", 0.0)
                                    if text or box:
                                        results.append({
                                            "text": text,
                                            "box": box,
                                            "score": score
                                        })
                    # 对于普通识别节点，直接处理text_results
                    else:
                        text_results = reco_detail.raw_detail.get("text_results", [])
                        if isinstance(text_results, list):
                            for result in text_results:
                                if isinstance(result, dict):
                                    text = result.get("text", "")
                                    box = result.get("box", [])
                                    score = result.get("score", 0.0)
                                    results.append({
                                        "text": text,
                                        "box": box,
                                        "score": score
                                    })
                        # 如果没有text_results字段，尝试获取最佳结果
                        elif not results and hasattr(reco_detail, "best_result"):
                            text = getattr(reco_detail.best_result, "text", "")
                            box = getattr(reco_detail, "box", [])
                            score = getattr(reco_detail.best_result, "score", 0.0)
                            results.append({
                                "text": text,
                                "box": box,
                                "score": score
                            })
                
                # 如果没有提取到结果，至少添加最佳结果
                if not results and hasattr(reco_detail, "best_result"):
                    text = getattr(reco_detail.best_result, "text", "")
                    box = getattr(reco_detail, "box", [])
                    score = getattr(reco_detail.best_result, "score", 0.0)
                    results.append({
                        "text": text,
                        "box": box,
                        "score": score
                    })
                
                logger.debug(f"RecognitionResultsArray: 识别成功，获取到 {len(results)} 个结果")
                return CustomRecognition.AnalyzeResult(
                    box=reco_detail.box,
                    detail={
                        "results": results
                    },
                )
            else:
                return None

        except Exception as e:
            logger.exception(f"RecognitionResultsArray识别失败: {e}")
            return None
            