#!/usr/bin/env python3
"""
感知驱动智能规划系统演示

本示例演示：
1. 感知驱动的任务规划 - 结合实时环境数据生成计划
2. CoT推理 - 可追溯的决策链
3. 多轮对话 - 指令澄清、执行确认、进度汇报
4. 响应式重规划 - 感知变化自动触发计划调整

运行方式：
    cd /media/yangyuhui/CODES1/Brain
    python3 examples/perception_driven_demo.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


# ============================================================
# 模拟用户交互回调
# ============================================================

class MockUserInterface:
    """模拟用户交互界面"""
    
    def __init__(self, auto_responses: dict = None):
        self.auto_responses = auto_responses or {}
        self.interaction_log = []
    
    async def user_callback(self, prompt: str, options: list) -> str:
        """模拟用户响应"""
        self.interaction_log.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "options": options
        })
        
        print(f"\n{'='*60}")
        print(f"🤖 系统: {prompt}")
        if options:
            print(f"   选项: {options}")
        
        # 自动响应逻辑
        for keyword, response in self.auto_responses.items():
            if keyword in prompt:
                print(f"👤 用户(自动): {response}")
                return response
        
        # 默认响应
        if options:
            response = options[0]
        elif "确认" in prompt or "继续" in prompt:
            response = "确认"
        else:
            response = "继续"
        
        print(f"👤 用户(自动): {response}")
        return response


# ============================================================
# 模拟传感器数据生成器
# ============================================================

class MockSensorDataGenerator:
    """模拟传感器数据生成器"""
    
    def __init__(self):
        self.tick = 0
        self.scenario = "normal"
    
    def set_scenario(self, scenario: str):
        """设置场景：normal, obstacle_appear, target_move, low_battery"""
        self.scenario = scenario
        logger.info(f"传感器场景切换为: {scenario}")
    
    def generate(self) -> dict:
        """生成传感器数据"""
        self.tick += 1
        
        base_data = {
            "gps": {
                "data": {
                    "latitude": 39.9042 + self.tick * 0.0001,
                    "longitude": 116.4074,
                    "altitude": 50 + self.tick
                }
            },
            "imu": {
                "data": {
                    "orientation": {"yaw": 45, "pitch": 0, "roll": 0}
                }
            },
            "battery": 100 - self.tick * 2,
            "detections": []
        }
        
        # 根据场景添加不同的检测结果
        if self.scenario == "obstacle_appear" and self.tick > 2:
            base_data["detections"].append({
                "id": "obstacle_1",
                "type": "obstacle",
                "x": 30, "y": 10, "z": 50,
                "confidence": 0.95
            })
        
        if self.scenario == "target_move" and self.tick > 3:
            base_data["detections"].append({
                "id": "target_1",
                "type": "person",
                "x": 50 + self.tick * 2, "y": 20, "z": 0,
                "confidence": 0.88,
                "is_target": True
            })
        
        if self.scenario == "low_battery":
            base_data["battery"] = max(15, 30 - self.tick * 3)
        
        return base_data


# ============================================================
# 演示场景
# ============================================================

async def demo_basic_perception_planning():
    """演示1: 基础感知驱动规划"""
    print("\n" + "="*70)
    print("📌 演示1: 基础感知驱动规划")
    print("="*70)
    
    from brain.cognitive.world_model import WorldModel
    from brain.cognitive.reasoning.cot_engine import CoTEngine
    from brain.cognitive.reasoning.reasoning_result import ReasoningMode
    from brain.cognitive.dialogue.dialogue_manager import DialogueManager
    
    # 创建组件
    world_model = WorldModel()
    cot_engine = CoTEngine()
    
    user_interface = MockUserInterface({
        "哪个": "东边的建筑"
    })
    dialogue = DialogueManager(user_callback=user_interface.user_callback)
    dialogue.start_session("demo_basic")
    
    # 模拟传感器数据
    sensor_data = {
        "gps": {"data": {"latitude": 39.9, "longitude": 116.4, "altitude": 50}},
        "imu": {"data": {"orientation": {"yaw": 45}}},
        "battery": 85,
        "detections": [
            {"id": "building_1", "type": "building", "x": 100, "y": 50, "z": 0, "confidence": 0.9},
            {"id": "tree_1", "type": "tree", "x": 30, "y": 10, "z": 0, "confidence": 0.85, "is_obstacle": True}
        ]
    }
    
    # 更新世界模型
    changes = world_model.update_from_perception(sensor_data)
    print(f"\n📡 感知更新: 检测到 {len(changes)} 个变化")
    
    # 获取规划上下文
    context = world_model.get_context_for_planning()
    print(f"\n🌍 规划上下文:")
    print(context.to_prompt_context())
    
    # 模拟模糊指令
    command = "去那边看看"
    print(f"\n💬 用户指令: {command}")
    
    # 检测到模糊，请求澄清
    ambiguities = [{"aspect": "位置", "question": "请问是哪个方向？", "options": ["东边的建筑", "西边的树"]}]
    clarification = await dialogue.clarify_ambiguous_command(command, ambiguities, context.to_prompt_context())
    print(f"\n✅ 澄清结果: {clarification['clarified_command']}")
    
    # CoT推理
    print(f"\n🧠 开始CoT推理...")
    reasoning = await cot_engine.reason(
        query=f"规划任务: {clarification['clarified_command']}",
        context={
            "obstacles": len(context.obstacles),
            "targets": len(context.targets),
            "battery_level": context.battery_level,
            "constraints": context.constraints
        },
        mode=ReasoningMode.PLANNING
    )
    
    print(f"\n📋 推理结果:")
    print(f"   复杂度: {reasoning.complexity.value}")
    print(f"   置信度: {reasoning.confidence:.2f}")
    print(f"   决策: {reasoning.decision}")
    print(f"   建议: {reasoning.suggestion}")
    print(f"\n   推理链摘要:")
    print(reasoning.get_chain_summary())
    
    dialogue.end_session()
    print("\n✅ 演示1完成")


async def demo_perception_driven_replan():
    """演示2: 感知驱动的重规划"""
    print("\n" + "="*70)
    print("📌 演示2: 感知驱动的重规划")
    print("="*70)
    
    from brain.cognitive.world_model import WorldModel, ChangeType
    from brain.cognitive.cot_engine import CoTEngine, ReasoningMode
    from brain.cognitive.monitoring.perception_monitor import PerceptionMonitor
    from brain.cognitive.dialogue_manager import DialogueManager
    
    # 创建组件
    world_model = WorldModel()
    cot_engine = CoTEngine()
    
    user_interface = MockUserInterface({
        "确认": "确认执行",
        "障碍": "确认绕行"
    })
    dialogue = DialogueManager(user_callback=user_interface.user_callback)
    dialogue.start_session("demo_replan")
    
    perception_monitor = PerceptionMonitor(world_model)
    sensor_gen = MockSensorDataGenerator()
    
    # 设置回调
    replan_events = []
    
    async def on_replan(event):
        replan_events.append(event)
        print(f"\n⚠️ 触发重规划: {event.change.description}")
    
    async def on_confirm(event) -> bool:
        result = await dialogue.request_confirmation(
            action=f"处理变化: {event.change.description}",
            reason=event.trigger.description,
            details=event.change.data
        )
        return result
    
    perception_monitor.set_replan_callback(on_replan)
    perception_monitor.set_confirmation_callback(on_confirm)
    
    print("\n🚀 开始任务执行模拟...")
    
    # 模拟执行循环
    for step in range(5):
        print(f"\n--- 执行步骤 {step + 1} ---")
        
        # 第3步出现障碍物
        if step == 2:
            sensor_gen.set_scenario("obstacle_appear")
        
        # 获取传感器数据
        sensor_data = sensor_gen.generate()
        print(f"   电池: {sensor_data['battery']}%")
        print(f"   检测到物体: {len(sensor_data['detections'])}个")
        
        # 处理传感器更新
        events = await perception_monitor.process_sensor_update(sensor_data)
        
        if events:
            for event in events:
                print(f"   📢 事件: {event.change.description} (动作: {event.action.value})")
                
                # 如果需要重规划
                if event.action.value in ["replan", "confirm_replan"]:
                    # 使用CoT评估
                    reasoning = await cot_engine.reason(
                        query="环境变化，是否需要调整计划？",
                        context={
                            "changes": event.change.description,
                            "change_data": event.change.data
                        },
                        mode=ReasoningMode.REPLANNING
                    )
                    
                    print(f"\n   🧠 CoT评估:")
                    print(f"      决策: {reasoning.decision}")
                    print(f"      置信度: {reasoning.confidence:.2f}")
        
        await asyncio.sleep(0.1)
    
    dialogue.end_session()
    
    print(f"\n📊 统计:")
    print(f"   触发的重规划事件: {len(replan_events)}")
    print(f"   感知监控状态: {perception_monitor.get_status()}")
    
    print("\n✅ 演示2完成")


async def demo_multi_turn_dialogue():
    """演示3: 多轮对话交互"""
    print("\n" + "="*70)
    print("📌 演示3: 多轮对话交互")
    print("="*70)
    
    from brain.cognitive.dialogue_manager import DialogueManager
    from brain.cognitive.world_model import WorldModel
    
    # 创建组件
    world_model = WorldModel()
    
    # 设置自动响应
    auto_responses = {
        "东边": "东边的A点",
        "确认": "确认",
        "暂停": "继续",
        "前方出现": "确认绕行",
        "完成": "好的"
    }
    
    user_interface = MockUserInterface(auto_responses)
    dialogue = DialogueManager(user_callback=user_interface.user_callback)
    dialogue.start_session("demo_dialogue")
    
    print("\n💬 开始多轮对话演示...")
    
    # 场景1: 指令澄清
    print("\n--- 场景1: 指令澄清 ---")
    clarification = await dialogue.clarify_ambiguous_command(
        command="去拍照",
        ambiguities=[{
            "aspect": "位置",
            "question": "去哪里拍照？",
            "options": ["东边的A点", "西边的B点", "当前位置"]
        }],
        world_context="当前位置(0,0,50)，东边100米有A点，西边80米有B点"
    )
    print(f"   澄清结果: {clarification}")
    
    # 场景2: 执行确认
    print("\n--- 场景2: 执行确认 ---")
    confirmed = await dialogue.request_confirmation(
        action="起飞并飞向A点",
        reason="这是任务的第一步",
        details={"目标高度": "100米", "预计时间": "30秒"}
    )
    print(f"   确认结果: {confirmed}")
    
    # 场景3: 进度汇报
    print("\n--- 场景3: 进度汇报 ---")
    adjustment = await dialogue.report_progress(
        status="飞行中",
        progress_percent=50,
        current_operation="goto",
        world_state_summary="高度80米，距离目标50米",
        allow_adjustment=True
    )
    print(f"   用户调整: {adjustment}")
    
    # 场景4: 错误报告
    print("\n--- 场景4: 错误报告 ---")
    choice = await dialogue.report_error(
        error="前方检测到移动物体",
        operation="goto",
        suggestions=["绕行", "悬停等待", "中止任务"],
        allow_choice=True
    )
    print(f"   用户选择: {choice}")
    
    # 场景5: 信息通知
    print("\n--- 场景5: 信息通知 ---")
    await dialogue.send_information("任务完成！共拍摄5张照片。")
    
    # 显示对话历史
    print("\n📜 对话历史:")
    for i, msg in enumerate(dialogue.get_conversation_history()):
        print(f"   {i+1}. [{msg['role']}] {msg['content'][:50]}...")
    
    dialogue.end_session()
    print("\n✅ 演示3完成")


async def demo_cot_reasoning():
    """演示4: CoT链式思维推理"""
    print("\n" + "="*70)
    print("📌 演示4: CoT链式思维推理")
    print("="*70)
    
    from brain.cognitive.reasoning.cot_engine import CoTEngine
    from brain.cognitive.reasoning.reasoning_result import ReasoningMode, ComplexityLevel
    
    cot_engine = CoTEngine()
    
    # 测试不同复杂度的任务
    test_cases = [
        {
            "query": "向前飞10米",
            "context": {"obstacles": 0, "battery_level": 90},
            "expected_complexity": ComplexityLevel.SIMPLE
        },
        {
            "query": "去东边拍照，注意避开障碍物",
            "context": {"obstacles": 3, "battery_level": 60, "constraints": ["避开禁飞区"]},
            "expected_complexity": ComplexityLevel.MODERATE
        },
        {
            "query": "搜索区域内的可疑目标，拍照记录，实时汇报",
            "context": {
                "obstacles": 5, 
                "targets": 2, 
                "battery_level": 40,
                "constraints": ["避开禁飞区", "保持通信"],
                "recent_changes": ["新发现目标", "天气变化"]
            },
            "expected_complexity": ComplexityLevel.COMPLEX
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- 测试案例 {i+1} ---")
        print(f"   任务: {case['query']}")
        
        # 评估复杂度
        complexity = cot_engine.assess_complexity(case['query'], case['context'])
        print(f"   评估复杂度: {complexity.value} (预期: {case['expected_complexity'].value})")
        
        # 执行推理
        result = await cot_engine.reason(
            query=case['query'],
            context=case['context'],
            mode=ReasoningMode.PLANNING
        )
        
        print(f"   实际复杂度: {result.complexity.value}")
        print(f"   置信度: {result.confidence:.2f}")
        print(f"   推理步骤数: {len(result.chain)}")
        print(f"   决策: {result.decision[:100]}...")
        print(f"   建议: {result.suggestion[:100]}...")
    
    print("\n✅ 演示4完成")


async def demo_full_integration():
    """演示5: 完整集成演示"""
    print("\n" + "="*70)
    print("📌 演示5: 完整集成演示 - 感知驱动的智能任务执行")
    print("="*70)
    
    from brain.cognitive.world_model import WorldModel
    from brain.cognitive.reasoning.cot_engine import CoTEngine
    from brain.cognitive.reasoning.reasoning_result import ReasoningMode
    from brain.cognitive.dialogue.dialogue_manager import DialogueManager
    from brain.cognitive.monitoring.perception_monitor import PerceptionMonitor
    from brain.llm.cot_prompts import CoTPrompts
    
    # 创建所有组件
    world_model = WorldModel()
    cot_engine = CoTEngine()
    cot_prompts = CoTPrompts()
    perception_monitor = PerceptionMonitor(world_model)
    
    user_interface = MockUserInterface({
        "确认": "确认",
        "东边": "东边50米处的目标点",
        "障碍": "确认绕行",
        "继续": "继续执行"
    })
    dialogue = DialogueManager(user_callback=user_interface.user_callback)
    
    sensor_gen = MockSensorDataGenerator()
    
    # 开始对话会话
    dialogue.start_session("integration_demo")
    
    print("\n🎯 模拟完整任务流程:")
    print("   1. 用户下达指令")
    print("   2. 系统请求澄清")
    print("   3. CoT推理生成计划")
    print("   4. 执行过程中感知变化")
    print("   5. 自动触发重规划")
    print("   6. 多轮对话确认")
    print("   7. 任务完成汇报")
    
    # Step 1: 用户指令
    user_command = "去那边的目标点执行侦察任务"
    print(f"\n📝 Step 1 - 用户指令: {user_command}")
    
    # Step 2: 获取感知数据
    print(f"\n📡 Step 2 - 获取感知数据...")
    sensor_data = sensor_gen.generate()
    sensor_data["detections"] = [
        {"id": "target_1", "type": "poi", "x": 50, "y": 0, "z": 0, "is_target": True, "confidence": 0.9},
        {"id": "obstacle_1", "type": "tree", "x": 25, "y": 5, "z": 0, "is_obstacle": True, "confidence": 0.85}
    ]
    
    world_model.update_from_perception(sensor_data)
    planning_context = world_model.get_context_for_planning()
    print(f"   感知状态: {world_model.get_summary()}")
    
    # Step 3: 指令澄清
    print(f"\n💬 Step 3 - 指令澄清...")
    clarification = await dialogue.clarify_ambiguous_command(
        command=user_command,
        ambiguities=[{
            "aspect": "位置",
            "question": "请确认目标位置",
            "options": ["东边50米处的目标点", "北边30米处的兴趣点"]
        }],
        world_context=planning_context.to_prompt_context()
    )
    clarified_command = clarification.get("clarified_command", user_command)
    
    # Step 4: CoT规划
    print(f"\n🧠 Step 4 - CoT规划...")
    planning_prompt = cot_prompts.build_planning_prompt(
        task_description=clarified_command,
        perception_context=planning_context.to_prompt_context(),
        available_operations="takeoff, goto, hover, scan_area, capture_image, return_to_home, land"
    )
    
    planning_reasoning = await cot_engine.reason(
        query=f"规划任务: {clarified_command}",
        context={
            "perception": planning_context.to_prompt_context(),
            "obstacles": len(planning_context.obstacles),
            "targets": len(planning_context.targets),
            "battery_level": planning_context.battery_level
        },
        mode=ReasoningMode.PLANNING
    )
    
    print(f"   规划推理完成:")
    print(f"   - 复杂度: {planning_reasoning.complexity.value}")
    print(f"   - 置信度: {planning_reasoning.confidence:.2f}")
    print(f"   - 决策: {planning_reasoning.decision}")
    
    # Step 5: 请求执行确认
    print(f"\n✅ Step 5 - 请求确认...")
    confirmed = await dialogue.request_confirmation(
        action="执行规划的任务",
        reason=f"基于CoT推理，置信度{planning_reasoning.confidence:.0%}",
        details={"操作数": "5", "预计时间": "120秒"}
    )
    
    if confirmed:
        print("\n🚀 Step 6 - 开始执行...")
        
        # 模拟执行过程
        operations = ["takeoff", "goto", "hover", "scan_area", "capture_image"]
        
        for i, op in enumerate(operations):
            progress = (i + 1) / len(operations) * 100
            
            # 更新感知
            if i == 2:
                # 模拟执行中出现新障碍
                sensor_gen.set_scenario("obstacle_appear")
            
            sensor_data = sensor_gen.generate()
            changes = world_model.update_from_perception(sensor_data)
            
            # 检查是否有显著变化
            significant_changes = world_model.detect_significant_changes()
            
            if significant_changes:
                print(f"\n   ⚠️ 检测到环境变化!")
                for change in significant_changes:
                    print(f"      - {change.description}")
                
                # CoT评估是否需要重规划
                replan_reasoning = await cot_engine.reason(
                    query="环境变化，是否需要调整计划？",
                    context={
                        "changes": [c.description for c in significant_changes],
                        "current_operation": op,
                        "progress": f"{progress:.0f}%"
                    },
                    mode=ReasoningMode.REPLANNING
                )
                
                print(f"   🧠 重规划评估: {replan_reasoning.decision}")
                
                if "replan" in replan_reasoning.decision.lower() or "调整" in replan_reasoning.decision:
                    await dialogue.report_and_confirm(
                        message=f"检测到环境变化，建议调整计划",
                        suggestion=replan_reasoning.suggestion
                    )
            
            # 汇报进度
            if i % 2 == 0:
                await dialogue.report_progress(
                    status=f"执行中 - {op}",
                    progress_percent=progress,
                    current_operation=op,
                    world_state_summary=f"电池{sensor_data['battery']}%",
                    allow_adjustment=False
                )
            
            print(f"   ✓ 完成操作: {op} ({progress:.0f}%)")
            await asyncio.sleep(0.1)
        
        # 任务完成
        print(f"\n🎉 Step 7 - 任务完成!")
        await dialogue.send_information(
            "✅ 侦察任务完成！\n"
            "- 拍摄照片: 3张\n"
            "- 扫描面积: 100平方米\n"
            "- 发现目标: 1个\n"
            "- 重规划次数: 1次"
        )
    
    # 显示统计
    print(f"\n📊 任务统计:")
    print(f"   对话轮次: {len(dialogue.get_conversation_history())}")
    print(f"   推理次数: {len(cot_engine.reasoning_history)}")
    print(f"   世界模型变化: {len(world_model.change_history)}")
    
    dialogue.end_session()
    print("\n✅ 演示5完成")


async def main():
    """主函数"""
    print("="*70)
    print("🧠 感知驱动智能规划系统 - 功能演示")
    print("="*70)
    print("""
本演示展示系统的核心能力：
1. 基础感知驱动规划 - 结合环境数据的智能规划
2. 感知驱动重规划 - 环境变化自动触发计划调整  
3. 多轮对话交互 - 澄清/确认/汇报
4. CoT链式思维推理 - 可追溯的决策过程
5. 完整集成演示 - 端到端任务执行流程
""")
    
    try:
        # 演示1: 基础感知驱动规划
        await demo_basic_perception_planning()
        
        # 演示2: 感知驱动重规划
        await demo_perception_driven_replan()
        
        # 演示3: 多轮对话
        await demo_multi_turn_dialogue()
        
        # 演示4: CoT推理
        await demo_cot_reasoning()
        
        # 演示5: 完整集成
        await demo_full_integration()
        
        print("\n" + "="*70)
        print("🎉 所有演示完成！")
        print("="*70)
        print("""
关键特性总结：
✅ 感知与规划深度融合 - 每次规划都结合实时感知数据
✅ 响应式重规划 - 环境变化自动触发，而非被动等待失败
✅ 多轮对话能力 - 支持澄清、确认、汇报三种对话模式
✅ CoT可追溯推理 - 所有决策都有推理链记录
✅ 自适应复杂度 - 简单任务快速执行，复杂任务深度推理
""")
        
    except Exception as e:
        logger.error(f"演示执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    # 运行演示
    asyncio.run(main())

