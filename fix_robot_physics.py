#!/usr/bin/env python3
"""
快速修复 Isaac Sim 中机器人的物理属性
在 Isaac Sim 中通过 Script Editor 运行此脚本
"""

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni


def fix_robot_physics(robot_prim_path="/NovaCarter"):
    """修复机器人物理属性"""
    
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    
    if not robot_prim.IsValid():
        print(f"❌ 未找到机器人: {robot_prim_path}")
        print("尝试查找机器人...")
        
        # 列出场景中的所有 Xform
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Xformable):
                prim_name = prim.GetName().lower()
                if 'nova' in prim_name or 'carter' in prim_name or 'robot' in prim_name:
                    print(f"找到可能的机器人: {prim.GetPath()}")
        
        return False
    
    print(f"✓ 找到机器人: {robot_prim_path}")
    
    # 1. 添加刚体属性
    UsdPhysics.RigidBodyAPI.Apply(robot_prim)
    rigid_body = UsdPhysics.RigidBodyAPI(robot_prim)
    rigid_body.CreateMassAttr().Set(10.0)  # 设置质量
    rigid_body.CreateEnabledAttr().Set(True)
    print("✓ 添加刚体属性")
    
    # 2. 为机器人主体添加碰撞体
    for prim in robot_prim.GetAllChildren():
        prim_name = prim.GetName().lower()
        
        # 跳过轮子和关节，只处理车身
        if 'wheel' in prim_name or 'joint' in prim_name or 'link' in prim_name:
            continue
        
        # 检查是否已经有碰撞体
        existing_collider = UsdPhysics.CollisionAPI(prim)
        if existing_collider:
            continue
        
        # 添加碰撞体
        UsdPhysics.CollisionAPI.Apply(prim)
        collision = UsdPhysics.CollisionAPI(prim)
        collision.CreateEnabledAttr().Set(True)
        print(f"✓ 为 {prim.GetName()} 添加碰撞体")
    
    # 3. 为轮子添加碰撞体和驱动
    for prim in robot_prim.GetAllChildren():
        prim_name = prim.GetName().lower()
        
        if 'wheel' in prim_name:
            # 添加碰撞体
            if not UsdPhysics.CollisionAPI(prim):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.CollisionAPI(prim).CreateEnabledAttr().Set(True)
                print(f"✓ 为轮子 {prim.GetName()} 添加碰撞体")
            
            # 添加驱动（如果是关节）
            for child in prim.GetAllChildren():
                if child.IsA(UsdPhysics.RevoluteJoint) or child.IsA(UsdPhysics.PrismaticJoint):
                    joint = child
                    PhysxSchema.PhysxDriveAPI.Apply(joint, "angular")
                    drive = PhysxSchema.PhysxDriveAPI.Get(joint, "angular")
                    drive.CreateMaxForceAttr().Set(1000.0)
                    drive.CreateDampingAttr().Set(0.0)
                    print(f"✓ 为关节 {joint.GetName()} 添加驱动")
    
    # 4. 调整机器人位置到地面上方
    xform = UsdGeom.Xformable(robot_prim)
    xformable = UsdGeom.Xformable(robot_prim)
    
    # 获取当前位置
    translate = xform.GetTranslateAttr()
    current_pos = translate.Get()
    
    # 设置 Z 为 0.5（在地面上方）
    new_pos = Gf.Vec3d(current_pos[0], current_pos[1], 0.5)
    translate.Set(new_pos)
    print(f"✓ 调整位置: z=0.5")
    
    # 5. 重置旋转
    rotate = xform.GetRotateXYZAttr()
    rotate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    print("✓ 重置旋转")
    
    print("\n✅ 物理修复完成!")
    print("现在按 Play 按钮测试，机器人应该不会掉下去")
    
    return True


def fix_ground_physics(ground_prim_path="/World/defaultPrim"):
    """修复地面物理属性"""
    
    stage = omni.usd.get_context().get_stage()
    
    # 尝试找到地面
    ground_prim = stage.GetPrimAtPath(ground_prim_path)
    
    # 如果默认路径不对，尝试查找
    if not ground_prim.IsValid():
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                prim_name = prim.GetName().lower()
                if 'plane' in prim_name or 'ground' in prim_name or 'floor' in prim_name:
                    ground_prim = prim
                    print(f"找到地面: {prim.GetPath()}")
                    break
    
    if not ground_prim.IsValid():
        print("⚠️  未找到地面，跳过")
        return False
    
    # 添加静态刚体
    if not UsdPhysics.RigidBodyAPI(ground_prim):
        UsdPhysics.RigidBodyAPI.Apply(ground_prim)
        UsdPhysics.RigidBodyAPI(ground_prim).CreateEnabledAttr().Set(True)
        print("✓ 为地面添加刚体属性")
    
    # 添加碰撞体
    if not UsdPhysics.CollisionAPI(ground_prim):
        UsdPhysics.CollisionAPI.Apply(ground_prim)
        UsdPhysics.CollisionAPI(ground_prim).CreateEnabledAttr().Set(True)
        print("✓ 为地面添加碰撞体")
    
    return True


def create_simple_robot(name="MyRobot"):
    """创建一个简单的机器人（如果没有可用的）"""
    
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.rotations import euler_angles_to_quat
    
    # 添加一个简单的差速驱动机器人
    usd_path = "omniverse://localhost/NVIDIA/Assets/Isaac/2022.2.1/Isaac/Robots/Wheeled_Robots/diff_drive/diff_drive.usd"
    
    try:
        prim = add_reference_to_stage(usd_path, f"/World/{name}")
        print(f"✓ 创建新机器人: {name}")
        
        # 设置位置
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
        print("✓ 设置初始位置")
        
        return True
    except Exception as e:
        print(f"❌ 创建机器人失败: {e}")
        return False


# 主函数
def main():
    """主函数"""
    print("="*60)
    print("Isaac Sim 机器人物理修复工具")
    print("="*60)
    print()
    
    # 尝试修复 Nova Carter
    print("1. 尝试修复 Nova Carter...")
    if not fix_robot_physics("/NovaCarter"):
        print("   Nova Carter 不存在或路径不对")
        
        # 尝试其他可能的路径
        possible_paths = [
            "/World/NovaCarter",
            "/NovaCarter_01",
            "/robot",
            "/World/robot"
        ]
        
        for path in possible_paths:
            print(f"   尝试路径: {path}")
            if fix_robot_physics(path):
                break
        else:
            print("\n❌ 无法找到 Nova Carter")
            print("\n是否创建一个简单的测试机器人? (输入 y/n)")
            # 注意：在 Isaac Script Editor 中无法获取用户输入
            # 需要手动决定是否运行下面的代码
            
            # 取消注释以下代码来创建新机器人
            # create_simple_robot("TestRobot")
    
    print("\n2. 修复地面...")
    fix_ground_physics()
    
    print("\n" + "="*60)
    print("修复完成!")
    print("="*60)
    print("\n下一步:")
    print("1. 点击 Play 按钮")
    print("2. 观察机器人是否停在地面")
    print("3. 如果还掉下去，增加机器人的 Z 坐标到 1.0")
    print()


# 在 Isaac Sim Script Editor 中运行此脚本
if __name__ == "__main__":
    main()






