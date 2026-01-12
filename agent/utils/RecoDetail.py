from typing import List, Tuple, Dict, Any, Optional
from maa.context import Context
from utils import logger


class RecoDetail:
    """
    识别详情处理工具类
    """

    @staticmethod
    def rect_to_dict(rect) -> Optional[Dict[str, int]]:
        """
        将 Rect 对象转换为可序列化的字典
        """
        if not rect:
            return None
        return {
            "x": getattr(rect, "x", 0),
            "y": getattr(rect, "y", 0),
            "width": getattr(rect, "width", 0),
            "height": getattr(rect, "height", 0),
        }

    @staticmethod
    def reco_detail_to_dict(reco) -> Optional[Dict[str, Any]]:
        """
        将 RecognitionDetail 对象转换为可序列化的字典
        """
        if not reco:
            return None
        return {
            "hit": getattr(reco, "hit", False),
            "box": RecoDetail.rect_to_dict(getattr(reco, "box", None)),
            "best_result": (
                {
                    "text": (
                        getattr(reco.best_result, "text", "")
                        if hasattr(reco, "best_result")
                        else ""
                    )
                }
                if hasattr(reco, "best_result")
                else None
            ),
            "raw_detail": getattr(reco, "raw_detail", None),
            "text": getattr(reco, "text", ""),
        }

    @staticmethod
    def extract_text_with_coordinates(raw_detail) -> List[Tuple[int, int, str]]:
        """
        从识别详情中提取文本和对应的 x、y 坐标
        """
        text_with_xy = []

        for item in raw_detail:
            logger.info(f"处理项目内容: {item}")

            # 处理字典类型的情况
            if isinstance(item, dict):
                RecoDetail._extract_from_dict_item(item, text_with_xy)
            # 处理对象类型的情况
            elif hasattr(item, "detail"):
                RecoDetail._extract_from_object_item(item, text_with_xy)

        return text_with_xy

    @staticmethod
    def _extract_from_dict_item(
        item: Dict[str, Any], text_with_xy: List[Tuple[int, int, str]]
    ) -> None:
        """
        从字典类型的项目中提取文本和 x、y 坐标
        """
        if "detail" in item:
            detail = item["detail"]
            if isinstance(detail, dict) and "all" in detail:
                all_items = detail["all"]
                if isinstance(all_items, list):
                    for all_item in all_items:
                        if (
                            isinstance(all_item, dict)
                            and "text" in all_item
                            and "box" in all_item
                        ):
                            box = all_item["box"]
                            if isinstance(box, list) and len(box) >= 2:
                                x = box[0]  # 获取 x 坐标
                                y = box[1]  # 获取 y 坐标
                                text_with_xy.append((x, y, all_item["text"]))
                                logger.info(
                                    f"找到带坐标的文本: {all_item['text']}，x={x}，y={y}"
                                )

    @staticmethod
    def _extract_from_object_item(
        item, text_with_xy: List[Tuple[int, int, str]]
    ) -> None:
        """
        从对象类型的项目中提取文本和 x、y 坐标
        """
        logger.info("处理对象类型")
        detail = getattr(item, "detail", {})
        if hasattr(detail, "all"):
            all_items = getattr(detail, "all", [])
            if isinstance(all_items, list):
                for all_item in all_items:
                    # 处理对象类型的 all_item
                    if hasattr(all_item, "text") and hasattr(all_item, "box"):
                        box = getattr(all_item, "box", [])
                        if isinstance(box, list) and len(box) >= 2:
                            x = box[0]  # 获取 x 坐标
                            y = box[1]  # 获取 y 坐标
                            text = getattr(all_item, "text", "")
                            text_with_xy.append((x, y, text))
                            logger.info(f"找到带坐标的文本: {text}，x={x}，y={y}")
                    # 兼容字典类型的 all_item
                    elif (
                        isinstance(all_item, dict)
                        and "text" in all_item
                        and "box" in all_item
                    ):
                        box = all_item["box"]
                        if isinstance(box, list) and len(box) >= 2:
                            x = box[0]  # 获取 x 坐标
                            y = box[1]  # 获取 y 坐标
                            text_with_xy.append((x, y, all_item["text"]))
                            logger.info(
                                f"找到带坐标的文本 (字典): {all_item['text']}，x={x}，y={y}"
                            )

    @staticmethod
    def split_by_y_threshold(
        text_with_xy: List[Tuple[int, int, str]],
    ) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
        """
        按 y 坐标阈值 300 分为我方（y较高）和敌方（y较低）两个数组
        """
        if not text_with_xy:
            return [], []

        threshold_y = 300
        logger.info(f"判断敌我阈值 y 坐标: {threshold_y}")

        enemy_texts = []  # 敌方（y较低）
        player_texts = []  # 我方（y较高）

        for x, y, text in text_with_xy:
            if y < threshold_y:
                enemy_texts.append((x, y, text))
                logger.info(f"添加到敌方: {text} (x={x}, y={y})")
            else:
                player_texts.append((x, y, text))
                logger.info(f"添加到我方: {text} (x={x}, y={y})")

        return enemy_texts, player_texts
