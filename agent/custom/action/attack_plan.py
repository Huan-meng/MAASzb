from maa.custom_action import CustomAction
from maa.context import Context
from maa.agent.agent_server import AgentServer
from utils import RecoDetail
from utils import logger
from typing import List, Tuple, Dict, Any


@AgentServer.custom_action("GenerateAttackPlan")
class GenerateAttackPlan(CustomAction):
    """
    根据识别详情生成攻击方案，使用 MonsterBattleSolver 求解最优策略，并生成拖动方案。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        # 获取原始识别详情
        raw_detail = getattr(argv.reco_detail, "raw_detail", [])
        logger.info(f"原始识别详情长度: {len(raw_detail)}")

        # 提取文本和对应的 x、y 坐标
        text_with_xy = RecoDetail.extract_text_with_coordinates(raw_detail)

        if not text_with_xy:
            logger.info("未找到文本")
            return CustomAction.RunResult(success=False)

        # 按 y 坐标阈值 300 分为我方和敌方
        enemy_texts, player_texts = RecoDetail.split_by_y_threshold(text_with_xy)

        # 处理敌方和我方的怪兽数据和位置
        enemy_monsters, enemy_positions = MonsterProcessor.process_monsters(enemy_texts)
        player_monsters, friend_positions = MonsterProcessor.process_monsters(
            player_texts
        )

        # 检查是否有有效的怪兽数据
        if not player_monsters:
            logger.info("无法提取有效的我方怪兽数据")
            return CustomAction.RunResult(success=False)

        # 获取敌方主战者生命值
        enemy_leader_hp = self._get_enemy_leader_hp(argv.reco_detail)

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
                # 构建 pipeline_override，覆盖对战中.json 中的随从交换节点
                pipeline_override = {
                    "随从交换": {"action": {"param": {"swipes": swipes}}}
                }
                result = context.run_action(
                    "随从交换", pipeline_override=pipeline_override
                )
                logger.info(f"MultiSwipe 动作执行结果: {result}")
            except Exception as e:
                logger.error(f"执行 MultiSwipe 动作时出错: {e}")
                return CustomAction.RunResult(success=False)
        else:
            logger.info("没有生成拖动方案")

        return CustomAction.RunResult(success=True)

    def _get_enemy_leader_hp(self, reco_detail) -> int:
        """
        获取敌方主战者生命值
        """
        enemy_leader_hp = 0
        try:
            # 尝试从识别结果中获取敌方主战者生命值
            # 假设 argv.reco_detail 中有 "识别对手主战血量" 的结果
            if hasattr(reco_detail, "识别对手主战血量"):
                leader_hp_result = getattr(reco_detail, "识别对手主战血量")
                if hasattr(leader_hp_result, "text"):
                    enemy_leader_hp = int(leader_hp_result.text)
                    logger.info(f"敌方主战者生命值: {enemy_leader_hp}")
                elif hasattr(leader_hp_result, "best_result"):
                    if hasattr(leader_hp_result.best_result, "text"):
                        enemy_leader_hp = int(leader_hp_result.best_result.text)
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
        # 每两个文本为一对（攻击力和生命值）
        for i in range(0, len(texts), 2):
            if i + 1 < len(texts):
                # 获取两个文本的坐标
                x1, y1, text1 = texts[i]
                x2, y2, text2 = texts[i + 1]

                # 计算平均坐标
                avg_x = (x1 + x2) // 2
                avg_y = (y1 + y2) // 2

                # 尝试转换为数字
                try:
                    atk = int(text1)
                    hp = int(text2)
                    monsters.append([atk, hp])
                    # 构建位置坐标 [x, y, width, height]
                    # 使用平均坐标作为中心点，宽度和高度设为默认值
                    positions.append([avg_x - 40, avg_y - 30, 80, 60])
                    logger.info(f"提取怪兽: 攻{atk}/血{hp}，坐标: ({avg_x}, {avg_y})")
                except ValueError:
                    logger.warning(f"无法转换为数字: {text1}, {text2}")
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

    ENEMY_LEADER_POSITION = [555, 4, 159, 80]

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

        # 检查我方总攻击力是否大于等于敌方主战者生命值
        if enemy_leader_hp > 0 and total_friend_atk >= enemy_leader_hp:
            logger.info("我方总攻击力大于等于敌方主战者生命值，直接攻击主战者")
            return AttackPlanGenerator._generate_attack_leader_swipes(
                player_monsters, friend_positions
            )

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
