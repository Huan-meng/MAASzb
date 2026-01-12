from maa.custom_action import CustomAction
from maa.context import Context
from maa.agent.agent_server import AgentServer
from utils import RecoDetail
from utils import logger
import json


@AgentServer.custom_action("GetRecoDetail")
class GetRecoDetail(CustomAction):
    """
    获取并打印前序识别详情。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        # 使用工具类处理识别详情
        reco_dict = RecoDetail.reco_detail_to_dict(argv.reco_detail)
        formatted_reco = json.dumps(reco_dict, ensure_ascii=False, indent=2)
        logger.info(f"识别详情文本: {formatted_reco}")

        return CustomAction.RunResult(success=True)
