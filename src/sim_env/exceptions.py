"""仿真环境异常类层次。"""


class RoadVehicleError(Exception):
    """仿真环境基础异常。"""

    pass


class ContinuityError(RoadVehicleError):
    """道路片段几何连续性验证失败。"""

    def __init__(self, segment_index: int, message: str):
        self.segment_index = segment_index
        super().__init__(f"片段 {segment_index} 连续性错误: {message}")


class RoadConfigError(RoadVehicleError):
    """道路配置无效。"""

    pass


class SerializationError(RoadVehicleError):
    """序列化/反序列化错误。"""

    pass
