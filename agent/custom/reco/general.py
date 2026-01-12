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
