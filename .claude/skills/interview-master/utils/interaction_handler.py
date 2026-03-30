"""
Interview Master - 交互处理器
实现用户点击选择器的响应逻辑和顿悟检测
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional

class InteractionHandler:
    """处理用户交互，包括按钮点击和顿悟检测"""

    def __init__(self, memory_path: str):
        """
        Args:
            memory_path: 记忆文件存储路径
        """
        self.memory_path = memory_path
        self.user_interactions_file = f"{memory_path}/user_interactions.json"
        self.eureka_moments_file = f"{memory_path}/eureka_moments.json"

        # 初始化交互记录文件
        self._initialize_interaction_files()

    def _initialize_interaction_files(self):
        """初始化交互记录文件"""
        # 用户交互记录
        try:
            with open(self.user_interactions_file, 'r') as f:
                self.interactions = json.load(f)
        except FileNotFoundError:
            self.interactions = {}
            self._save_json(self.user_interactions_file, self.interactions)

        # 顿悟时刻记录
        try:
            with open(self.eureka_moments_file, 'r') as f:
                self.eureka_moments = json.load(f)
        except FileNotFoundError:
            self.eureka_moments = {}
            self._save_json(self.eureka_moments_file, self.eureka_moments)

    def handle_button_click(self, user_id: str, concept: str, button_type: str,
                          current_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理用户点击交互按钮

        Args:
            user_id: 用户ID
            concept: 当前概念（如"positional_encoding"）
            button_type: 按钮类型（更简单些/深入数学/给我看代码/关联概念）
            current_content: 当前内容
            context: 上下文信息（用户画像、当前模板等）

        Returns:
            更新后的内容
        """
        # 记录用户交互
        self._log_interaction(user_id, concept, button_type, context)

        # 根据按钮类型生成新内容
        if button_type == "更简单些":
            return self._simplify_content(current_content, context)
        elif button_type == "深入数学":
            return self._deepen_math(current_content, context)
        elif button_type == "给我看代码":
            return self._switch_to_code(current_content, context)
        elif button_type == "关联到其他概念":
            return self._show_relations(current_content, context)
        elif button_type == "跳过这部分":
            return self._skip_and_mark_mastered(user_id, concept, context)
        else:
            return {"status": "error", "message": f"未知的按钮类型: {button_type}"}

    def _log_interaction(self, user_id: str, concept: str, button_type: str,
                        context: Dict[str, Any]):
        """记录用户交互到知识图谱"""
        if user_id not in self.interactions:
            self.interactions[user_id] = {}

        if concept not in self.interactions[user_id]:
            self.interactions[user_id][concept] = []

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "button_type": button_type,
            "context": {
                "template": context.get("template"),
                "math_tolerance": context.get("math_tolerance"),
                "mastery_score": context.get("mastery_score", 0)
            }
        }

        self.interactions[user_id][concept].append(interaction)
        self._save_json(self.user_interactions_file, self.interactions)

        # 更新知识图谱（记录学习偏好）
        self._update_knowledge_graph(user_id, concept, button_type, context)

    def _update_knowledge_graph(self, user_id: str, concept: str,
                               button_type: str, context: Dict[str, Any]):
        """更新知识图谱中的学习偏好"""
        kg_file = f"{self.memory_path}/knowledge_graphs/main.json"

        try:
            with open(kg_file, 'r') as f:
                kg = json.load(f)
        except FileNotFoundError:
            kg = {"concepts": {}, "user_patterns": {}}

        # 更新概念级别的学习偏好
        if concept not in kg["concepts"]:
            kg["concepts"][concept] = {}

        concept_data = kg["concepts"][concept]

        # 根据按钮点击调整推荐策略
        if button_type == "更简单些":
            concept_data["learning_difficulty"] = "high"
            concept_data["needs_simplification"] = True
        elif button_type == "深入数学":
            concept_data["math_tolerance"] = "high"
            concept_data["prefers_derivation"] = True
        elif button_type == "给我看代码":
            concept_data["code_preference"] = "high"
            concept_data["needs_implementation"] = True

        # 记录点击频率（用于检测常见困惑点）
        if "interaction_history" not in concept_data:
            concept_data["interaction_history"] = []

        concept_data["interaction_history"].append({
            "user_id": user_id,
            "button_type": button_type,
            "timestamp": datetime.now().isoformat()
        })

        # 如果超过50%用户点击"更简单些", 标记为抽象困难概念
        self._analyze_common_confusions(concept, concept_data)

        self._save_json(kg_file, kg)

    def _analyze_common_confusions(self, concept: str, concept_data: Dict[str, Any]):
        """分析常见困惑点"""
        interactions = concept_data.get("interaction_history", [])

        if len(interactions) < 5:  # 需要至少5次交互才能分析
            return

        # 统计各按钮点击频率
        from collections import Counter
        button_counts = Counter([i["button_type"] for i in interactions])

        # 检测模式
        if button_counts.get("更简单些", 0) / len(interactions) > 0.5:
            concept_data["common_confusion_type"] = "abstract_concept"
            concept_data["teaching_recommendation"] = "优先使用类比和可视化"

        if button_counts.get("给我看代码", 0) / len(intersections) > 0.4:
            concept_data["common_confusion_type"] = "implementation_detail"
            concept_data["teaching_recommendation"] = "增加代码示例和实践案例"

    def detect_aha_moment(self, user_id: str, concept: str, user_input: str,
                         context: Dict[str, Any]) -> bool:
        """
        检测用户是否产生顿悟（Aha!时刻）

        Args:
            user_id: 用户ID
            concept: 当前概念
            user_input: 用户输入文本
            context: 上下文

        Returns:
            是否检测到顿悟
        """
        # 顿悟信号词（中文和英文）
        eureka_signals = [
            # 中文
            "啊哈", "哦！", "原来如此", "我懂了", "明白了",
            "恍然大悟", "茅塞顿开", "豁然开朗",
            # 英文
            "aha", "oh!", "i see", "got it", "eureka",
            "now i understand", "that makes sense"
        ]

        # 困惑信号词（说明之前不懂，现在懂了）
        confusion_to_clarity = [
            "之前不懂", "之前不明白", "现在懂了", "现在明白了",
            "之前一直困惑", "之前的疑问解开了"
        ]

        # 检测顿悟
        has_eureka = any(signal in user_input.lower() for signal in eureka_signals)
        has_clarity = any(signal in user_input for signal in confusion_to_clarity)

        if has_eureka or has_clarity:
            # 记录顿悟时刻
            self._record_eureka_moment(user_id, concept, user_input, context)
            return True

        return False

    def _record_eureka_moment(self, user_id: str, concept: str, user_input: str,
                             context: Dict[str, Any]):
        """记录顿悟时刻"""
        if user_id not in self.eureka_moments:
            self.eureka_moments[user_id] = {}

        if concept not in self.eureka_moments[user_id]:
            self.eureka_moments[user_id][concept] = []

        eureka_data = {
            "timestamp": datetime.now().isoformat(),
            "trigger_words": user_input,
            "understanding_summary": "",  # 用户后续会补充
            "context": {
                "template_used": context.get("template"),
                "explanation_method": context.get("last_explanation_method"),
                "depth_before": context.get("mastery_score", 0),
                "depth_after": min(context.get("mastery_score", 0) + 0.15, 1.0)  # 顿悟后掌握度提升
            }
        }

        self.eureka_moments[user_id][concept].append(eureka_data)
        self._save_json(self.eureka_moments_file, self.eureka_moments)

        # 更新知识图谱（记录有效的教学方法）
        self._update_teaching_effectiveness(concept, context)

    def _update_teaching_effectiveness(self, concept: str, context: Dict[str, Any]):
        """更新教学方法有效性数据"""
        kg_file = f"{self.memory_path}/knowledge_graphs/main.json"

        try:
            with open(kg_file, 'r') as f:
                kg = json.load(f)
        except FileNotFoundError:
            return

        if concept not in kg["concepts"]:
            return

        concept_data = kg["concepts"][concept]

        # 记录有效的教学触发条件
        if "effective_teaching_methods" not in concept_data:
            concept_data["effective_teaching_methods"] = []

        method = context.get("last_explanation_method", "unknown")
        concept_data["effective_teaching_methods"].append({
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "success_rate": "detected_eureka"
        })

        self._save_json(kg_file, kg)

    def get_refinement_suggestion(self, user_id: str, concept: str) -> Optional[str]:
        """
        根据用户的交互历史，获取后续讲解优化建议

        Returns:
            优化建议字符串，如果没有数据则返回None
        """
        # 检查用户的交互历史
        user_interactions = self.interactions.get(user_id, {})
        concept_interactions = user_interactions.get(concept, [])

        if not concept_interactions:
            return None

        # 分析交互模式
        button_types = [i["button_type"] for i in concept_interactions]

        if "更简单些" in button_types:
            return "用户在此概念上需要更多类比和可视化，建议优先使用生活场景讲解"

        if "给我看代码" in button_types:
            return "用户偏好代码实现，建议增加代码示例和工程细节"

        if "深入数学" in button_types:
            return "用户有数学背景，建议提供形式化定义和推导链路"

        return None

    def _simplify_content(self, current_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """简化内容，增加类比"""
        return {
            "status": "success",
            "action": "simplify",
            "message": "正在生成更简单的解释，添加更多生活类比...",
            "next_template": "intuition_constructive",
            "instruction": "请追加2-3个生活类比，使用更简单的语言，降低抽象层级"
        }

    def _deepen_math(self, current_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """深化数学内容"""
        return {
            "status": "success",
            "action": "deepen_math",
            "message": "正在添加数学推导和形式化定义...",
            "next_template": "hardcore_derivation",
            "instruction": "请插入LaTeX公式推导，展开边界条件分析"
        }

    def _switch_to_code(self, current_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """切换到代码实现"""
        return {
            "status": "success",
            "action": "show_code",
            "message": "切换到工程Checklist模式，提供完整实现...",
            "next_template": "engineering_checklist",
            "instruction": "请提供核心代码实现，包含关键注释和生产环境考虑"
        }

    def _show_relations(self, current_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """显示关联概念"""
        return {
            "status": "success",
            "action": "show_relations",
            "message": "展示知识图谱和相关概念...",
            "next_template": context.get("template"),  # 保持当前模板
            "instruction": "请展示知识图谱DAG，列出前置依赖和后续概念"
        }

    def _skip_and_mark_mastered(self, user_id: str, concept: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """跳过并标记为已掌握"""
        # 更新知识图谱
        self._update_mastery_score(user_id, concept, 0.9)  # 跳过假设已掌握

        return {
            "status": "success",
            "action": "mark_mastered",
            "message": "已标记为掌握，下次会推荐更高级的概念",
            "next_recommended": context.get("next_recommended", [])
        }

    def _update_mastery_score(self, user_id: str, concept: str, new_score: float):
        """更新掌握度评分"""
        kg_file = f"{self.memory_path}/knowledge_graphs/main.json"

        try:
            with open(kg_file, 'r') as f:
                kg = json.load(f)
        except FileNotFoundError:
            return

        if concept in kg.get("concepts", {}):
            kg["concepts"][concept]["mastery_score"] = new_score
            self._save_json(kg_file, kg)

    def _save_json(self, filepath: str, data: Dict[str, Any]):
        """保存JSON文件"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class FineGrainedEvaluator:
    """细粒度掌握度评估（5维度）"""

    def __init__(self):
        self.evaluation_dimensions = {
            "theoretical_understanding": {
                "description": "理论理解深度",
                "weight": 0.25,
                "indicators": ["能解释核心概念", "理解数学原理", "知道边界条件"]
            },
            "code_readiness": {
                "description": "代码实现能力",
                "weight": 0.25,
                "indicators": ["能写出伪代码", "知道PyTorch实现", "理解工程细节"]
            },
            "interview_defense": {
                "description": "面试防御能力",
                "weight": 0.20,
                "indicators": ["准备高频追问", "有实际项目经验", "知道工业界实践"]
            },
            "math_proof": {
                "description": "数学推导能力",
                "weight": 0.15,
                "indicators": ["理解公式推导", "能做复杂度分析", "了解前沿改进"]
            },
            "business_application": {
                "description": "业务应用能力",
                "weight": 0.15,
                "indicators": ["理解业务价值", "能做成本分析", "知道落地路径"]
            }
        }

    def evaluate_mastery(self, user_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        细粒度评估用户掌握度

        Args:
            user_performance: {
                "correct_answers": 8,
                "total_questions": 10,
                "can_explain_concept": True,
                "can_write_code": False,
                "has_practical_experience": False,
                "understands_math": True,
                "knows_business_value": False
            }

        Returns:
            5维度评分 + 总评分
        """
        scores = {}

        # 计算各维度分数
        qa_accuracy = user_performance.get("correct_answers", 0) / user_performance.get("total_questions", 1)

        scores["theoretical_understanding"] = min(qa_accuracy * 1.2 + 0.1, 1.0)

        scores["code_readiness"] = 0.7 if user_performance.get("can_write_code") else 0.3

        scores["interview_defense"] = 0.8 if user_performance.get("has_practical_experience") else 0.4

        scores["math_proof"] = 0.85 if user_performance.get("understands_math") else 0.2

        scores["business_application"] = 0.75 if user_performance.get("knows_business_value") else 0.25

        # 计算加权总分
        total_score = sum(scores[dim] * self.evaluation_dimensions[dim]["weight"]
                         for dim in scores)

        return {
            "total_score": round(total_score, 2),
            "breakdown": {dim: round(score, 2) for dim, score in scores.items()},
            "strengths": [dim for dim, score in scores.items() if score > 0.7],
            "weaknesses": [dim for dim, score in scores.items() if score < 0.5],
            "recommendations": self._generate_recommendations(scores)
        }

    def _generate_recommendations(self, scores: Dict[str, float]) -> list:
        """根据薄弱点生成学习建议"""
        recommendations = []

        if scores["code_readiness"] < 0.5:
            recommendations.append("增加代码实践，从伪代码开始逐步到完整实现")

        if scores["interview_defense"] < 0.5:
            recommendations.append("准备高频面试问题，积累项目实战经验")

        if scores["business_application"] < 0.5:
            recommendations.append("学习业务场景，理解技术落地的商业价值")

        if scores["math_proof"] < 0.5:
            recommendations.append("补充数学基础，理解公式背后的原理")

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 测试交互处理器
    handler = InteractionHandler("/tmp/interview_master_memory")

    # 测试顿悟检测
    test_input = "啊哈！原来位置编码是为了解决Attention的置换不变性！"
    is_eureka = handler.detect_aha_moment("user_001", "positional_encoding",
                                        test_input, {"template": "intuition"})
    print(f"检测到顿悟: {is_eureka}")

    # 测试细粒度评估
    evaluator = FineGrainedEvaluator()
    result = evaluator.evaluate_mastery({
        "correct_answers": 8,
        "total_questions": 10,
        "can_explain_concept": True,
        "can_write_code": False,  # 薄弱点
        "has_practical_experience": False,  # 薄弱点
        "understands_math": True,
        "knows_business_value": False  # 薄弱点
    })
    print("\n细粒度评估结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
