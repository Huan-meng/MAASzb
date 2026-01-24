from maa.custom_action import CustomAction
from maa.context import Context
from maa.agent.agent_server import AgentServer
from utils import RecoDetail
from utils import logger
from typing import List, Tuple, Dict, Any
import json


@AgentServer.custom_action("GenerateAttackPlan")
class GenerateAttackPlan(CustomAction):
    """
    根据识别详情生成攻击方案，使用 MonsterBattleSolver 求解最优策略，并生成拖动方案。
    """

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
            else:
                player_texts.append((x, y, text))

        return enemy_texts, player_texts

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        # 获取原始识别详情
        raw_detail = getattr(argv.reco_detail, "raw_detail", [])

        # 提取文本和对应的 x、y 坐标
        text_with_xy = RecoDetail.extract_text_with_coordinates(
            raw_detail["best"]["detail"]
        )

        if not text_with_xy:
            logger.info("未找到文本")
            return CustomAction.RunResult(success=False)

        # 按 y 坐标阈值 300 分为我方和敌方
        enemy_texts, player_texts = GenerateAttackPlan.split_by_y_threshold(
            text_with_xy
        )

        # 处理敌方和我方的怪兽数据和位置[936, 616, 47, 103]
        enemy_monsters, enemy_positions = MonsterProcessor.process_monsters(enemy_texts)
        player_monsters, friend_positions = MonsterProcessor.process_monsters(
            player_texts
        )

        # 检查是否有有效的怪兽数据
        if not player_monsters:
            logger.info("无法提取有效的我方怪兽数据")
            return CustomAction.RunResult(success=False)

        # 获取敌方主战者生命值
        enemy_leader_hp = self.get_enemy_leader_hp(context)

        # 生成攻击方案
        swipes = AttackPlanGenerator.generate_attack_plan(
            context,
            enemy_monsters,
            enemy_positions,
            player_monsters,
            friend_positions,
            enemy_leader_hp,
        )

        # 如果有拖动操作，执行 MultiSwipe 动作
        if swipes:
            logger.info(f"生成的拖动方案: {swipes}")
            # 使用 context 执行 MultiSwipe 动作
            try:
                # 构建 pipeline_override，覆盖对战中.json 中的多指滑动泛用型节点
                pipeline_override = {
                    "多指滑动泛用型": {"action": {"param": {"swipes": swipes}}}
                }
                result = context.run_action(
                    "多指滑动泛用型", pipeline_override=pipeline_override
                )

            except Exception as e:
                logger.error(f"执行 MultiSwipe 动作时出错: {e}")
                return CustomAction.RunResult(success=False)
        else:
            logger.info("没有生成拖动方案")

        return CustomAction.RunResult(success=True)

    def get_enemy_leader_hp(self, context: Context) -> int:
        """
        获取敌方主战者生命值
        """
        enemy_leader_hp = 0
        try:
            reco_detail = context.run_recognition(
                "识别对手主战血量", context.tasker.controller.cached_image
            )
            reco_dict = RecoDetail.reco_detail_to_dict(reco_detail)
            enemy_leader_hp = int(reco_dict["best_result"]["text"])
            logger.info(f"敌方主战者生命值: {enemy_leader_hp}")
        except Exception as e:
            logger.warning(f"获取敌方主战者生命值时出错: {e}")
        return enemy_leader_hp


class MonsterProcessor:
    """
    怪兽数据处理类
    """

    @staticmethod
    def process_monsters(
        texts: List[Tuple[int, int, str]],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        计算每个怪兽的平均坐标并生成怪兽数据
        """
        monsters = []
        positions = []

        # 清洗数组，过滤掉非数字的元素
        cleaned_texts = []
        for item in texts:
            x, y, text = item
            if text.isdigit():
                cleaned_texts.append(item)
            else:
                logger.warning(f"过滤非数字元素: {text}")

        if not cleaned_texts:
            return monsters, positions

        # 按 x 坐标排序
        sorted_texts = sorted(cleaned_texts, key=lambda item: item[0])
        logger.info(f"按 x 坐标排序后的文本: {sorted_texts}")

        # 合并x坐标距离小于15的数字
        merged_texts = []
        i = 0
        while i < len(sorted_texts):
            if i + 1 < len(sorted_texts):
                x1, y1, text1 = sorted_texts[i]
                x2, y2, text2 = sorted_texts[i + 1]
                if x2 - x1 < 15:
                    # 合并两个数字，计算平均坐标
                    merged_x = (x1 + x2) // 2
                    merged_y = (y1 + y2) // 2
                    merged_text = text1 + text2
                    merged_texts.append((merged_x, merged_y, merged_text))
                    logger.info(
                        f"合并数字: {text1} + {text2} = {merged_text}，坐标: ({merged_x}, {merged_y})"
                    )
                    i += 2
                else:
                    merged_texts.append(sorted_texts[i])
                    i += 1
            else:
                merged_texts.append(sorted_texts[i])
                i += 1
        logger.info(f"合并后的文本: {merged_texts}")

        # 按 x 坐标临近程度聚类
        clusters = []
        current_cluster = []
        min_threshold_x = 80  # x 坐标最小阈值
        max_threshold_x = 140  # x 坐标最大阈值
        logger.info(f"x 坐标阈值区间: {min_threshold_x} - {max_threshold_x}")

        for item in merged_texts:
            if not current_cluster:
                current_cluster.append(item)
            else:
                last_x = current_cluster[-1][0]
                current_x = item[0]
                if min_threshold_x <= current_x - last_x <= max_threshold_x:
                    current_cluster.append(item)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [item]
        if current_cluster:
            clusters.append(current_cluster)

        # 过滤出包含恰好两个元素的集群（生命值和攻击力）
        valid_clusters = [cluster for cluster in clusters if len(cluster) == 2]
        logger.info(f"有效怪兽集群数量: {len(valid_clusters)}")

        if not valid_clusters:
            logger.warning("没有找到有效的怪兽集群")
            return monsters, positions

        # 处理每个有效集群
        for cluster in valid_clusters:
            atk_x, atk_y, atk_text = cluster[0]
            hp_x, hp_y, hp_text = cluster[1]

            # 计算平均坐标
            avg_x = (atk_x + hp_x) // 2
            avg_y = (atk_y + hp_y) // 2

            # 尝试转换为数字
            try:
                atk = int(atk_text)
                hp = int(hp_text)
                monsters.append([atk, hp])
                # 构建位置坐标 [x, y, width, height]
                # 使用平均坐标作为中心点，宽度和高度设为默认值
                positions.append([avg_x - 33, avg_y - 79, 94, 73])
                logger.info(f"提取怪兽: 攻{atk}/血{hp}，坐标: ({avg_x}, {avg_y})")
            except ValueError:
                logger.warning(f"无法转换为数字: {atk_text}, {hp_text}")

        return monsters, positions

    @staticmethod
    def calculate_total_attack(monsters: List[List[int]]) -> int:
        """
        计算我方总攻击力
        """
        return sum(monster[0] for monster in monsters)


class AttackPlanGenerator:
    """
    攻击方案生成器
    """

    ENEMY_LEADER_POSITION = [578, 57, 121, 49]

    @staticmethod
    def generate_attack_plan(
        context: Context,
        enemy_monsters: List[List[int]],
        enemy_positions: List[List[int]],
        player_monsters: List[List[int]],
        friend_positions: List[List[int]],
        enemy_leader_hp: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        生成攻击方案
        """
        swipes = []

        # 检查是否只有我方怪兽没有敌方怪兽
        if not enemy_monsters and player_monsters:
            logger.info("只有我方怪兽，没有敌方怪兽，直接攻击主战者")
            return AttackPlanGenerator._generate_attack_leader_swipes(
                player_monsters, friend_positions
            )

        # 计算我方总攻击力
        total_friend_atk = MonsterProcessor.calculate_total_attack(player_monsters)
        logger.info(f"我方总攻击力: {total_friend_atk}")

        # # 检查我方总攻击力是否大于等于敌方主战者生命值
        # if enemy_leader_hp > 0 and total_friend_atk >= enemy_leader_hp:
        #     logger.info("我方总攻击力大于等于敌方主战者生命值，直接攻击主战者")
        #     return AttackPlanGenerator._generate_attack_leader_swipes(
        #         player_monsters, friend_positions
        #     )

        # 使用 MonsterBattleSolver 求解最优攻击策略
        solver = MonsterBattleSolver(enemy_monsters, player_monsters)
        enemy_killed, friend_dead, attack_plan = solver.solve()

        # 生成攻击拖动
        return AttackPlanGenerator._generate_attack_monsters_swipes(
            attack_plan, enemy_positions, friend_positions, player_monsters
        )

    @staticmethod
    def _generate_attack_leader_swipes(
        player_monsters: List[List[int]], friend_positions: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        生成攻击主战者的拖动方案
        """
        swipes = []
        for j in range(len(player_monsters)):
            if j < len(friend_positions):
                swipe = {
                    "starting": j * 500,  # 每个攻击间隔500ms
                    "begin": friend_positions[j],
                    "end": AttackPlanGenerator.ENEMY_LEADER_POSITION,
                    "duration": 300,
                    "end_hold": 100,
                }
                swipes.append(swipe)
                logger.info(f"生成攻击主战者拖动: 我方怪兽{j+1} -> 对手主战者")
        return swipes

    @staticmethod
    def _generate_attack_monsters_swipes(
        attack_plan: List[List[int]],
        enemy_positions: List[List[int]],
        friend_positions: List[List[int]],
        player_monsters: List[List[int]],
    ) -> List[Dict[str, Any]]:
        """
        生成攻击敌方怪兽的拖动方案
        """
        swipes = []
        used_monsters = set()

        # 生成攻击拖动
        for i, attackers in enumerate(attack_plan):
            if attackers and i < len(enemy_positions):
                for j, attacker_idx in enumerate(attackers):
                    if attacker_idx < len(friend_positions):
                        swipe = {
                            "starting": len(swipes) * 500,  # 每个攻击间隔500ms
                            "begin": friend_positions[attacker_idx],
                            "end": enemy_positions[i],
                            "duration": 300,
                            "end_hold": 100,
                        }
                        swipes.append(swipe)
                        used_monsters.add(attacker_idx)
                        logger.info(
                            f"生成攻击拖动: 我方怪兽{attacker_idx+1} -> 敌方怪兽{i+1}"
                        )

        # 生成不攻击的怪兽拖动到对手主战者
        for j in range(len(player_monsters)):
            if j not in used_monsters and j < len(friend_positions):
                swipe = {
                    "starting": len(swipes) * 500,  # 接在上一个攻击之后
                    "begin": friend_positions[j],
                    "end": AttackPlanGenerator.ENEMY_LEADER_POSITION,
                    "duration": 300,
                    "end_hold": 100,
                }
                swipes.append(swipe)
                logger.info(f"生成攻击主战者拖动: 我方怪兽{j+1} -> 对手主战者")

        return swipes


class MonsterBattleSolver:
    """
    怪兽战斗求解器
    """

    def __init__(
        self, enemy_monsters: List[List[int]], friend_monsters: List[List[int]]
    ):
        """
        初始化怪兽战斗求解器

        参数:
            enemy_monsters: 敌方怪兽列表，每个元素为[攻击力, 生命值]
            friend_monsters: 我方怪兽列表，每个元素为[攻击力, 生命值]
        """
        # 将输入转换为攻击力和生命值列表
        self.enemy_atk = [monster[0] for monster in enemy_monsters]
        self.enemy_hp = [monster[1] for monster in enemy_monsters]
        self.friend_atk = [monster[0] for monster in friend_monsters]
        self.friend_hp = [monster[1] for monster in friend_monsters]

        self.a = len(enemy_monsters)  # 敌方怪兽数量
        self.b = len(friend_monsters)  # 我方怪兽数量

        # 预计算：每个掩码对应的总攻击力
        self.attack_sum = [0] * (1 << self.b)
        for mask in range(1, 1 << self.b):
            # 找到最低位的1
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            # 递归计算：当前掩码攻击力 = 去掉最低位的掩码攻击力 + 当前怪兽攻击力
            self.attack_sum[mask] = self.attack_sum[mask ^ lsb] + self.friend_atk[idx]

        # 预计算：对于每个敌方怪兽，每个掩码会死亡的我方怪兽数量
        self.death_count = [[0] * (1 << self.b) for _ in range(self.a)]
        for i in range(self.a):
            for mask in range(1 << self.b):
                # 同样使用递归计算
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                prev_mask = mask ^ lsb
                self.death_count[i][mask] = self.death_count[i][prev_mask]
                # 如果敌方攻击力 >= 我方生命值，则我方怪兽死亡
                if self.enemy_atk[i] >= self.friend_hp[idx]:
                    self.death_count[i][mask] += 1

    def solve(self) -> Tuple[int, int, List[List[int]]]:
        """
        求解最优攻击策略

        返回:
            enemy_killed: 消灭的敌方怪兽数量
            friend_dead: 死亡的我方怪兽数量
            attack_plan: 攻击方案，每个元素是攻击对应敌方怪兽的我方怪兽索引列表
        """
        if self.a == 0 or self.b == 0:
            return 0, 0, [[] for _ in range(self.a)]

        # DP状态: dp[i][mask] = (消灭敌方数, 我方死亡数, 前驱状态, 攻击掩码)
        # 初始化所有状态为(-1, float('inf'), -1, -1)
        dp = [
            [(-1, float("inf"), -1, -1) for _ in range(1 << self.b)]
            for _ in range(self.a + 1)
        ]

        # 初始状态：没有考虑任何敌方怪兽，没有使用我方怪兽
        dp[0][0] = (0, 0, -1, -1)

        # 动态规划转移
        for i in range(self.a):
            for mask in range(1 << self.b):
                current_killed, current_dead, _, _ = dp[i][mask]

                # 无效状态，跳过
                if current_killed == -1:
                    continue

                # 获取dp[i+1][mask]的当前值
                next_killed, next_dead, _, _ = dp[i + 1][mask]

                # 选项1：不攻击敌方怪兽 i
                if self._better_state(
                    (current_killed, current_dead), (next_killed, next_dead)
                ):
                    dp[i + 1][mask] = (current_killed, current_dead, mask, 0)

                # 选项2：攻击敌方怪兽 i
                # 获取未使用的我方怪兽掩码
                unused_mask = ((1 << self.b) - 1) ^ mask

                # 枚举所有非空子集
                sub_mask = unused_mask
                while sub_mask > 0:
                    # 检查攻击力是否足够消灭敌方怪兽
                    if self.attack_sum[sub_mask] >= self.enemy_hp[i]:
                        new_mask = mask | sub_mask
                        new_killed = current_killed + 1
                        new_dead = current_dead + self.death_count[i][sub_mask]

                        # 获取dp[i+1][new_mask]的当前值
                        next_killed2, next_dead2, _, _ = dp[i + 1][new_mask]

                        if self._better_state(
                            (new_killed, new_dead), (next_killed2, next_dead2)
                        ):
                            dp[i + 1][new_mask] = (new_killed, new_dead, mask, sub_mask)

                    # 下一个子集
                    sub_mask = (sub_mask - 1) & unused_mask

        # 找到最优解
        best_killed = 0
        best_dead = float("inf")
        best_mask = 0

        for mask in range(1 << self.b):
            killed, dead, _, _ = dp[self.a][mask]
            if killed > best_killed or (killed == best_killed and dead < best_dead):
                best_killed = killed
                best_dead = dead
                best_mask = mask

        # 回溯构建攻击方案
        attack_plan = [[] for _ in range(self.a)]
        current_mask = best_mask

        for i in range(self.a, 0, -1):
            _, _, prev_mask, attack_mask = dp[i][current_mask]

            if attack_mask > 0:
                # 将掩码转换为怪兽索引
                monster_indices = []
                for j in range(self.b):
                    if attack_mask & (1 << j):
                        monster_indices.append(j)
                attack_plan[i - 1] = monster_indices

            current_mask = prev_mask

        return best_killed, best_dead, attack_plan

    def _better_state(self, state1: Tuple[int, int], state2: Tuple[int, int]) -> bool:
        """
        比较两个状态，返回state1是否优于state2
        优先比较消灭敌方数量，再比较我方死亡数量
        """
        killed1, dead1 = state1
        killed2, dead2 = state2

        # 如果state2是无效状态，则state1更好
        if killed2 == -1:
            return True

        if killed1 > killed2:
            return True
        elif killed1 == killed2 and dead1 < dead2:
            return True
        return False


@AgentServer.custom_action("GenerateEvolutionPlan")
class GenerateEvolutionPlan(CustomAction):
    """
    生成进化和超进化方案，从指定位置拖动到我方怪兽位置。
    """

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
            else:
                player_texts.append((x, y, text))

        return enemy_texts, player_texts

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        # 获取原始识别详情
        raw_detail = getattr(argv.reco_detail, "raw_detail", [])

        # 提取文本和对应的 x、y 坐标
        text_with_xy = RecoDetail.extract_text_with_coordinates(
            raw_detail["best"]["detail"]
        )

        if not text_with_xy:
            logger.info("未找到文本")
            return CustomAction.RunResult(success=False)

        # 按 y 坐标阈值 300 分为我方和敌方
        enemy_texts, player_texts = GenerateEvolutionPlan.split_by_y_threshold(
            text_with_xy
        )

        # 处理我方的怪兽数据和位置
        _, friend_positions = MonsterProcessor.process_monsters(player_texts)

        # 检查是否有有效的怪兽数据
        if not friend_positions:
            logger.info("无法提取有效的我方怪兽数据")
            return CustomAction.RunResult(success=False)

        # 生成进化和超进化方案
        swipes = EvolutionPlanGenerator.generate_evolution_plan(friend_positions)

        # 如果有拖动操作，执行 MultiSwipe 动作
        if swipes:
            logger.info(f"生成的进化和超进化拖动方案: {swipes}")
            # 使用 context 执行 MultiSwipe 动作
            try:
                # 构建 pipeline_override，覆盖对战中.json 中的多指滑动泛用型节点
                pipeline_override = {
                    "多指滑动泛用型": {"action": {"param": {"swipes": swipes}}}
                }
                result = context.run_action(
                    "多指滑动泛用型", pipeline_override=pipeline_override
                )
            except Exception as e:
                logger.error(f"执行 MultiSwipe 动作时出错: {e}")
                return CustomAction.RunResult(success=False)
        else:
            logger.info("没有生成进化和超进化拖动方案")

        return CustomAction.RunResult(success=True)


class EvolutionPlanGenerator:
    """
    进化和超进化方案生成器
    """

    EVOLUTION_POSITION = [540, 492, 44, 54]
    SUPER_EVOLUTION_POSITION = [700, 496, 36, 42]

    @staticmethod
    def generate_evolution_plan(
        friend_positions: List[List[int]],
    ) -> List[Dict[str, Any]]:
        """
        生成进化和超进化方案，依次从进化和超进化位置拖动到我方每个怪兽位置
        """
        swipes = []

        if not friend_positions:
            return swipes

        # 生成从进化位置到每个我方怪兽的拖动
        for i, position in enumerate(friend_positions):
            # 进化拖动
            evolution_swipe = {
                "starting": i * 1000,  # 每个进化间隔1000ms（进化+超进化）
                "begin": EvolutionPlanGenerator.EVOLUTION_POSITION,
                "end": position,
                "duration": 300,
                "end_hold": 100,
            }
            swipes.append(evolution_swipe)
            logger.info(f"生成进化拖动: 进化位置 -> 我方怪兽{i+1}")

            # 超进化拖动
            super_evolution_swipe = {
                "starting": i * 1000 + 500,  # 超进化在进化后500ms执行
                "begin": EvolutionPlanGenerator.SUPER_EVOLUTION_POSITION,
                "end": position,
                "duration": 300,
                "end_hold": 100,
            }
            swipes.append(super_evolution_swipe)
            logger.info(f"生成超进化拖动: 超进化位置 -> 我方怪兽{i+1}")

        return swipes
