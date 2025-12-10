# Python Types for Development

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

A comprehensive guide to Python data types with a focus on IoT applications. This repository serves as a reference for developers working with sensor data, device configurations, and IoT system development.

## 🚀 Features

- Complete overview of Python primitive and derived types
- IoT-specific examples and use cases
- Type annotation best practices
- Real-world sensor data handling examples
- Ready-to-use code snippets

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Type Categories](#type-categories)
  - [Primitive Types](#primitive-types)
  - [Derived Types](#derived-types)
  - [User-Defined Types](#user-defined-types)
  - [Type Aliases](#type-aliases)
  - [Function Types](#function-types)
- [IoT Applications](#iot-applications)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

link : https://www.youtube.com/watch?v=v8veitfmmDolist=PLu71SKxNbfoBsMugTFALhdLlZ5VOqCg2s&index=6

## 🚀 Quick Start

```python
# Basic sensor data handling
temperature = 25.5  # float
device_id = 123     # int
is_active = True    # bool

# Collection of sensor readings
sensor_readings = [25.5, 26.0, 25.8, 27.1]
device_config = {"id": 123, "status": "active", "location": "room_1"}

# Type annotations for better code clarity
from typing import Dict, List

def process_sensor_data(readings: List[float]) -> Dict[str, float]:
    return {
        "average": sum(readings) / len(readings),
        "max": max(readings),
        "min": min(readings)
    }

result = process_sensor_data(sensor_readings)
print(result)  # {'average': 26.1, 'max': 27.1, 'min': 25.5}
```

## 📊 Type Categories

### Primitive Types

Basic, built-in types perfect for representing fundamental sensor values and device states.

| Type | Description | IoT Example |
|------|-------------|-------------|
| `int` | Integer numbers | Device ID, Port numbers |
| `float` | Decimal numbers | Temperature, Humidity readings |
| `str` | Text strings | Device names, Status messages |
| `bool` | True/False values | Device state, Alarm status |
| `complex` | Complex numbers | Signal processing (rare) |

```python
# Sensor data examples
temperature = 23.7        # float
humidity = 65            # int (can be float too)
device_name = "TempSensor01"  # str
is_online = True         # bool
```

### Derived Types

Built-in collection types for managing complex IoT data structures.

```python
# List: Mutable sequence of sensor readings
hourly_temps = [22.1, 23.5, 24.8, 26.2, 25.9]

# Tuple: Immutable coordinate or fixed data pair
gps_location = (40.7128, -74.0060)  # (latitude, longitude)

# Dictionary: Device configuration and metadata
device_config = {
    "id": "SENSOR_001",
    "type": "temperature",
    "location": "greenhouse",
    "calibration_offset": 0.5
}

# Set: Unique device identifiers
active_devices = {"TEMP_01", "HUM_02", "PRESS_03"}
```

### User-Defined Types

Custom classes for modeling complex IoT entities and systems.

```python
from datetime import datetime
from typing import Optional

class IoTDevice:
    """Represents an IoT device with sensor capabilities."""
    
    def __init__(self, device_id: str, device_type: str):
        self.device_id = device_id
        self.device_type = device_type
        self.is_online = False
        self.last_reading: Optional[float] = None
        self.timestamp: Optional[datetime] = None
    
    def update_reading(self, value: float) -> None:
        """Update device with new sensor reading."""
        self.last_reading = value
        self.timestamp = datetime.now()
        self.is_online = True
    
    def get_status(self) -> dict:
        """Return current device status."""
        return {
            "id": self.device_id,
            "type": self.device_type,
            "online": self.is_online,
            "last_reading": self.last_reading,
            "last_update": self.timestamp
        }

# Usage
temp_sensor = IoTDevice("TEMP_001", "temperature")
temp_sensor.update_reading(24.3)
print(temp_sensor.get_status())
```

### Type Aliases

Simplify complex type definitions for better code readability.

```python
from typing import Dict, List, Tuple

# Type aliases for common IoT data structures
SensorReading = Tuple[str, float, datetime]  # (sensor_id, value, timestamp)
DeviceConfig = Dict[str, str]                # Configuration dictionary
ReadingHistory = List[SensorReading]         # List of historical readings

# Usage with type hints
def analyze_readings(history: ReadingHistory) -> Dict[str, float]:
    """Analyze historical sensor readings."""
    if not history:
        return {}
    
    values = [reading[1] for reading in history]
    return {
        "count": len(values),
        "average": sum(values) / len(values),
        "max": max(values),
        "min": min(values)
    }
```

### Function Types

Functions and lambdas for data processing and device control logic.

```python
from typing import Callable

# Regular function for data validation
def validate_temperature(temp: float) -> bool:
    """Validate temperature reading is within acceptable range."""
    return -50.0 <= temp <= 100.0

# Lambda for quick data transformation
celsius_to_fahrenheit = lambda c: (c * 9/5) + 32

# Higher-order function for sensor data processing
def apply_sensor_filter(
    readings: List[float], 
    filter_func: Callable[[float], bool]
) -> List[float]:
    """Apply a filter function to sensor readings."""
    return [reading for reading in readings if filter_func(reading)]

# Usage
temperatures = [22.5, 150.0, 23.1, -60.0, 24.8]  # Some invalid readings
valid_temps = apply_sensor_filter(temperatures, validate_temperature)
print(valid_temps)  # [22.5, 23.1, 24.8]
```

## 🌐 IoT Applications

Python's type system excels in IoT development for:

### 📡 **Data Collection & Processing**
- Type-safe sensor data validation
- Structured device configuration management
- Efficient data aggregation and analysis

### 🔧 **Device Management**
- Object-oriented device modeling
- Configuration validation with type hints
- State management with enums and classes

### 📊 **Real-time Analytics**
- Type-annotated data pipelines
- Statistical analysis with NumPy integration
- Time-series data handling

### 🚨 **Alert Systems**
- Boolean logic for threshold monitoring
- Event-driven architectures with typed callbacks
- Message queue integration (MQTT, etc.)

## 💡 Examples

### Complete IoT Sensor System

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum

class SensorType(Enum):
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"

@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: SensorType
    value: float
    timestamp: datetime
    unit: str

class IoTHub:
    """Central hub for managing multiple IoT devices."""
    
    def __init__(self):
        self.devices: List[IoTDevice] = []
        self.readings: List[SensorReading] = []
    
    def add_device(self, device: IoTDevice) -> None:
        self.devices.append(device)
    
    def log_reading(self, reading: SensorReading) -> None:
        self.readings.append(reading)
        print(f"📊 {reading.sensor_type.value}: {reading.value}{reading.unit}")
    
    def get_latest_readings(self, sensor_type: SensorType) -> List[SensorReading]:
        return [r for r in self.readings if r.sensor_type == sensor_type]

# Usage example
hub = IoTHub()
temp_sensor = IoTDevice("TEMP_001", "temperature")
hub.add_device(temp_sensor)

# Simulate sensor reading
reading = SensorReading(
    sensor_id="TEMP_001",
    sensor_type=SensorType.TEMPERATURE,
    value=23.5,
    timestamp=datetime.now(),
    unit="°C"
)
hub.log_reading(reading)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Python Official Documentation](https://docs.python.org/3/)
- [Python Typing Module](https://docs.python.org/3/library/typing.html)
- [IoT with Python Resources](https://realpython.com/python-iot/)

## ⭐ Support

If you found this helpful, please consider giving it a star ⭐️

---

**Made with ❤️ for the IoT development community**