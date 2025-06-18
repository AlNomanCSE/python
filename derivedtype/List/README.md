# Python Lists Complete Guide 📋

A comprehensive guide to Python lists covering declaration, manipulation, and practical applications. Perfect for beginners and intermediate developers looking to master Python list operations.

## 🚀 Quick Start

```python
# Create a list
sensor_readings = [23.5, 24.1, 22.8, 25.0]

# Add elements
sensor_readings.append(26.2)
sensor_readings.extend([27.1, 24.9])

# Remove elements
sensor_readings.remove(22.8)
deleted_value = sensor_readings.pop()

print(sensor_readings)  # [23.5, 24.1, 25.0, 26.2, 27.1]
```

## 📋 Table of Contents

- [List Declaration](#list-declaration)
- [Adding Elements](#adding-elements)
- [Removing Elements](#removing-elements)
- [List Operations](#list-operations)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)


## 📝 List Declaration

### 1. Empty Lists

```python
# Method 1: Using square brackets (recommended)
empty_list = []

# Method 2: Using list() constructor
empty_list = list()

# Method 3: List comprehension (empty)
empty_list = [x for x in []]
```

### 2. Lists with Initial Values

```python
# Numbers
temperatures = [23.5, 24.1, 22.8, 25.0, 26.2]
device_ids = [1, 2, 3, 4, 5]

# Strings
device_names = ["TempSensor01", "HumSensor02", "PressSensor03"]
statuses = ["online", "offline", "maintenance"]

# Mixed data types
sensor_data = ["TempSensor01", 23.5, True, "online"]

# Boolean values
device_status = [True, False, True, True, False]
```

### 3. Lists from Other Iterables

```python
# From string
char_list = list("hello")  # ['h', 'e', 'l', 'l', 'o']

# From tuple
tuple_data = (1, 2, 3, 4, 5)
list_data = list(tuple_data)  # [1, 2, 3, 4, 5]

# From range
numbers = list(range(1, 6))  # [1, 2, 3, 4, 5]
even_numbers = list(range(0, 11, 2))  # [0, 2, 4, 6, 8, 10]
```

### 4. List Comprehensions

```python
# Basic comprehension
squares = [x**2 for x in range(1, 6)]  # [1, 4, 9, 16, 25]

# With condition
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]  # [4, 16, 36, 64, 100]

# From existing list
celsius_temps = [20, 25, 30, 35]
fahrenheit_temps = [(c * 9/5) + 32 for c in celsius_temps]  # [68.0, 77.0, 86.0, 95.0]
```

### 5. Nested Lists (2D Arrays)

```python
# 2D list for sensor grid
sensor_grid = [
    [23.5, 24.1, 22.8],
    [25.0, 26.2, 24.9],
    [23.1, 24.5, 25.8]
]

# List of dictionaries
devices = [
    {"id": 1, "name": "TempSensor01", "value": 23.5},
    {"id": 2, "name": "HumSensor02", "value": 65.2},
    {"id": 3, "name": "PressSensor03", "value": 1013.25}
]
```

## ➕ Adding Elements

### 1. append() - Add Single Element to End

```python
sensor_readings = [23.5, 24.1, 22.8]

# Add single element
sensor_readings.append(25.0)
print(sensor_readings)  # [23.5, 24.1, 22.8, 25.0]

# Add different data types
device_names = ["Sensor01"]
device_names.append("Sensor02")
device_names.append("Sensor03")
print(device_names)  # ['Sensor01', 'Sensor02', 'Sensor03']
```

### 2. extend() - Add Multiple Elements

```python
temperatures = [23.5, 24.1]

# Extend with another list
new_temps = [25.0, 26.2, 24.9]
temperatures.extend(new_temps)
print(temperatures)  # [23.5, 24.1, 25.0, 26.2, 24.9]

# Extend with any iterable
temperatures.extend(range(27, 30))
print(temperatures)  # [23.5, 24.1, 25.0, 26.2, 24.9, 27, 28, 29]
```

### 3. insert() - Add Element at Specific Position

```python
devices = ["Sensor01", "Sensor02", "Sensor04"]

# Insert at specific index
devices.insert(2, "Sensor03")
print(devices)  # ['Sensor01', 'Sensor02', 'Sensor03', 'Sensor04']

# Insert at beginning
devices.insert(0, "MainSensor")
print(devices)  # ['MainSensor', 'Sensor01', 'Sensor02', 'Sensor03', 'Sensor04']
```

### 4. List Concatenation

```python
temp_readings = [23.5, 24.1, 22.8]
humidity_readings = [65.2, 67.1, 63.5]

# Using + operator
all_readings = temp_readings + humidity_readings
print(all_readings)  # [23.5, 24.1, 22.8, 65.2, 67.1, 63.5]

# Using += operator (modifies original)
temp_readings += [25.0, 26.2]
print(temp_readings)  # [23.5, 24.1, 22.8, 25.0, 26.2]
```

### 5. Unpacking and Adding

```python
readings = [23.5, 24.1]
new_data = [25.0, 26.2, 24.9]

# Using unpacking
combined = [*readings, *new_data, 27.1, 28.3]
print(combined)  # [23.5, 24.1, 25.0, 26.2, 24.9, 27.1, 28.3]
```

## ➖ Removing Elements

### 1. remove() - Remove First Occurrence

```python
temperatures = [23.5, 24.1, 22.8, 24.1, 25.0]

# Remove first occurrence of value
temperatures.remove(24.1)
print(temperatures)  # [23.5, 22.8, 24.1, 25.0]

# Handle value not found
try:
    temperatures.remove(30.0)  # Value not in list
except ValueError:
    print("Value not found in list")
```

### 2. pop() - Remove by Index

```python
sensor_readings = [23.5, 24.1, 22.8, 25.0, 26.2]

# Remove and return last element
last_reading = sensor_readings.pop()
print(last_reading)  # 26.2
print(sensor_readings)  # [23.5, 24.1, 22.8, 25.0]

# Remove and return specific index
first_reading = sensor_readings.pop(0)
print(first_reading)  # 23.5
print(sensor_readings)  # [24.1, 22.8, 25.0]
```

### 3. del Statement - Remove by Index or Slice

```python
devices = ["Sensor01", "Sensor02", "Sensor03", "Sensor04", "Sensor05"]

# Delete single element
del devices[1]
print(devices)  # ['Sensor01', 'Sensor03', 'Sensor04', 'Sensor05']

# Delete slice
del devices[1:3]
print(devices)  # ['Sensor01', 'Sensor05']

# Delete entire list
del devices
```

### 4. clear() - Remove All Elements

```python
sensor_data = [23.5, 24.1, 22.8, 25.0]

# Clear all elements
sensor_data.clear()
print(sensor_data)  # []
```

### 5. List Comprehension for Conditional Removal

```python
temperatures = [18.5, 23.5, 24.1, 15.2, 25.0, 12.8]

# Keep only temperatures above 20°C
filtered_temps = [temp for temp in temperatures if temp > 20.0]
print(filtered_temps)  # [23.5, 24.1, 25.0]

# Remove specific values
readings = [23.5, -999, 24.1, -999, 25.0]  # -999 indicates error
clean_readings = [r for r in readings if r != -999]
print(clean_readings)  # [23.5, 24.1, 25.0]
```

## 🔧 List Operations

### Accessing Elements

```python
sensors = ["TempSensor", "HumSensor", "PressSensor", "LightSensor"]

# Positive indexing
print(sensors[0])    # TempSensor
print(sensors[2])    # PressSensor

# Negative indexing
print(sensors[-1])   # LightSensor
print(sensors[-2])   # PressSensor

# Slicing
print(sensors[1:3])  # ['HumSensor', 'PressSensor']
print(sensors[:2])   # ['TempSensor', 'HumSensor']
print(sensors[2:])   # ['PressSensor', 'LightSensor']
```

### List Methods

```python
readings = [23.5, 24.1, 22.8, 24.1, 25.0]

# Count occurrences
count = readings.count(24.1)  # 2

# Find index
index = readings.index(25.0)  # 4

# Sort list
readings.sort()  # Modifies original
print(readings)  # [22.8, 23.5, 24.1, 24.1, 25.0]

# Reverse list
readings.reverse()  # Modifies original
print(readings)  # [25.0, 24.1, 24.1, 23.5, 22.8]

# Copy list
readings_copy = readings.copy()
```

## 🎯 Use Cases

### 1. IoT Sensor Data Collection

```python
class SensorDataCollector:
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.timestamps = []
    
    def add_reading(self, temp, humidity, timestamp):
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.timestamps.append(timestamp)
    
    def get_average_temperature(self):
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def remove_outliers(self, threshold=5.0):
        avg_temp = self.get_average_temperature()
        self.temperature_readings = [
            temp for temp in self.temperature_readings 
            if abs(temp - avg_temp) <= threshold
        ]

# Usage
collector = SensorDataCollector()
collector.add_reading(23.5, 65.2, "2024-01-01 10:00:00")
collector.add_reading(24.1, 67.1, "2024-01-01 10:05:00")
collector.add_reading(35.0, 45.0, "2024-01-01 10:10:00")  # Outlier

print(f"Average before filtering: {collector.get_average_temperature():.1f}°C")
collector.remove_outliers()
print(f"Average after filtering: {collector.get_average_temperature():.1f}°C")
```

### 2. Device Management System

```python
class DeviceManager:
    def __init__(self):
        self.online_devices = []
        self.offline_devices = []
        self.maintenance_devices = []
    
    def add_device(self, device_id, status="offline"):
        if status == "online":
            self.online_devices.append(device_id)
        elif status == "maintenance":
            self.maintenance_devices.append(device_id)
        else:
            self.offline_devices.append(device_id)
    
    def change_device_status(self, device_id, new_status):
        # Remove from all lists
        for device_list in [self.online_devices, self.offline_devices, self.maintenance_devices]:
            if device_id in device_list:
                device_list.remove(device_id)
                break
        
        # Add to appropriate list
        self.add_device(device_id, new_status)
    
    def get_all_devices(self):
        return self.online_devices + self.offline_devices + self.maintenance_devices
    
    def get_status_report(self):
        return {
            "online": len(self.online_devices),
            "offline": len(self.offline_devices),
            "maintenance": len(self.maintenance_devices),
            "total": len(self.get_all_devices())
        }

# Usage
manager = DeviceManager()
manager.add_device("TEMP_001", "online")
manager.add_device("HUM_002", "online")
manager.add_device("PRESS_003", "offline")

print(manager.get_status_report())
```

### 3. Data Processing Pipeline

```python
def process_sensor_data(raw_readings):
    """Process raw sensor data through multiple stages"""
    
    # Stage 1: Remove invalid readings (None, negative values)
    valid_readings = [r for r in raw_readings if r is not None and r >= 0]
    
    # Stage 2: Remove outliers (beyond 3 standard deviations)
    if len(valid_readings) > 1:
        mean = sum(valid_readings) / len(valid_readings)
        variance = sum((x - mean) ** 2 for x in valid_readings) / len(valid_readings)
        std_dev = variance ** 0.5
        
        filtered_readings = [
            r for r in valid_readings 
            if abs(r - mean) <= 3 * std_dev
        ]
    else:
        filtered_readings = valid_readings
    
    # Stage 3: Apply calibration (example: add 0.5°C offset)
    calibrated_readings = [r + 0.5 for r in filtered_readings]
    
    # Stage 4: Round to 1 decimal place
    final_readings = [round(r, 1) for r in calibrated_readings]
    
    return final_readings

# Usage
raw_data = [23.2, 24.5, None, 23.8, -5.0, 24.1, 100.0, 23.9, 24.2]
processed_data = process_sensor_data(raw_data)
print(f"Raw: {raw_data}")
print(f"Processed: {processed_data}")
```

### 4. Configuration Management

```python
class ConfigManager:
    def __init__(self):
        self.sensor_configs = []
        self.alert_thresholds = []
        self.enabled_features = []
    
    def add_sensor_config(self, sensor_type, settings):
        config = {"type": sensor_type, "settings": settings}
        self.sensor_configs.append(config)
    
    def update_sensor_config(self, sensor_type, new_settings):
        for config in self.sensor_configs:
            if config["type"] == sensor_type:
                config["settings"].update(new_settings)
                break
    
    def remove_sensor_config(self, sensor_type):
        self.sensor_configs = [
            config for config in self.sensor_configs 
            if config["type"] != sensor_type
        ]
    
    def get_config_by_type(self, sensor_type):
        for config in self.sensor_configs:
            if config["type"] == sensor_type:
                return config
        return None

# Usage
config_mgr = ConfigManager()
config_mgr.add_sensor_config("temperature", {"min": -10, "max": 50, "unit": "°C"})
config_mgr.add_sensor_config("humidity", {"min": 0, "max": 100, "unit": "%"})

temp_config = config_mgr.get_config_by_type("temperature")
print(temp_config)
```

### 5. Time Series Data Analysis

```python
def analyze_time_series(timestamps, values):
    """Analyze time series sensor data"""
    
    if not timestamps or not values or len(timestamps) != len(values):
        return {}
    
    # Combine timestamps and values
    time_series = list(zip(timestamps, values))
    
    # Sort by timestamp
    time_series.sort(key=lambda x: x[0])
    
    # Extract sorted values
    sorted_values = [value for _, value in time_series]
    
    # Calculate statistics
    analysis = {
        "count": len(sorted_values),
        "min": min(sorted_values),
        "max": max(sorted_values),
        "average": sum(sorted_values) / len(sorted_values),
        "range": max(sorted_values) - min(sorted_values)
    }
    
    # Calculate trend (simple)
    if len(sorted_values) >= 2:
        first_half = sorted_values[:len(sorted_values)//2]
        second_half = sorted_values[len(sorted_values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        if avg_second > avg_first:
            analysis["trend"] = "increasing"
        elif avg_second < avg_first:
            analysis["trend"] = "decreasing"
        else:
            analysis["trend"] = "stable"
    
    return analysis

# Usage
timestamps = ["10:00", "10:05", "10:10", "10:15", "10:20"]
temperatures = [23.5, 24.1, 24.8, 25.2, 25.9]

result = analyze_time_series(timestamps, temperatures)
print(result)
```

## 🚀 Performance Tips

### 1. Efficient List Operations

```python
# ✅ Good: Use list comprehension
squared = [x**2 for x in range(1000)]

# ❌ Avoid: Using append in loops for large datasets
squared = []
for x in range(1000):
    squared.append(x**2)

# ✅ Good: Use extend for multiple elements
readings.extend(new_readings)

# ❌ Avoid: Multiple append calls
for reading in new_readings:
    readings.append(reading)
```

### 2. Memory Optimization

```python
# For large datasets, consider using generators
def sensor_readings_generator():
    for i in range(1000000):
        yield i * 0.1  # Simulate sensor reading

# Use only when needed
readings = sensor_readings_generator()
```

### 3. Searching and Filtering

```python
# ✅ Good: Use 'in' for membership testing
if "TEMP_001" in device_list:
    process_device("TEMP_001")

# ✅ Good: Use set for frequent lookups
device_set = set(device_list)
if "TEMP_001" in device_set:  # O(1) vs O(n)
    process_device("TEMP_001")
```

## 🎯 Best Practices

### 1. Naming Conventions

```python
# ✅ Good: Descriptive names
temperature_readings = [23.5, 24.1, 22.8]
device_names = ["TempSensor01", "HumSensor02"]
online_devices = ["DEV_001", "DEV_002"]

# ❌ Avoid: Generic names
data = [23.5, 24.1, 22.8]
items = ["TempSensor01", "HumSensor02"]
list1 = ["DEV_001", "DEV_002"]
```

### 2. Type Hints

```python
from typing import List, Optional

def process_readings(readings: List[float]) -> List[float]:
    """Process sensor readings and return filtered results"""
    return [r for r in readings if r > 0]

def find_device(devices: List[str], device_id: str) -> Optional[str]:
    """Find device in list, return None if not found"""
    return device_id if device_id in devices else None
```

### 3. Error Handling

```python
def safe_list_operations(data_list, index, value):
    """Demonstrate safe list operations"""
    try:
        # Safe indexing
        if 0 <= index < len(data_list):
            return data_list[index]
        
        # Safe removal
        if value in data_list:
            data_list.remove(value)
        
        return data_list
    
    except (IndexError, ValueError) as e:
        print(f"List operation error: {e}")
        return data_list
```

## 📊 Examples

### Complete IoT Data Management System

```python
from datetime import datetime
from typing import List, Dict, Optional

class IoTDataManager:
    """Complete IoT data management system using lists"""
    
    def __init__(self):
        self.devices: List[Dict] = []
        self.sensor_readings: List[Dict] = []
        self.alerts: List[Dict] = []
        self.maintenance_log: List[Dict] = []
    
    def register_device(self, device_id: str, device_type: str, location: str):
        """Register a new IoT device"""
        device = {
            "id": device_id,
            "type": device_type,
            "location": location,
            "status": "offline",
            "registered_at": datetime.now().isoformat()
        }
        self.devices.append(device)
        print(f"✅ Device {device_id} registered successfully")
    
    def add_sensor_reading(self, device_id: str, value: float, unit: str):
        """Add a new sensor reading"""
        reading = {
            "device_id": device_id,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat()
        }
        self.sensor_readings.append(reading)
        
        # Check for alerts
        self._check_alerts(device_id, value)
    
    def _check_alerts(self, device_id: str, value: float):
        """Check if reading triggers any alerts"""
        # Example: Temperature alert
        if value > 30.0:  # High temperature threshold
            alert = {
                "device_id": device_id,
                "type": "high_temperature",
                "value": value,
                "message": f"High temperature detected: {value}°C",
                "timestamp": datetime.now().isoformat()
            }
            self.alerts.append(alert)
            print(f"🚨 ALERT: {alert['message']}")
    
    def get_device_readings(self, device_id: str) -> List[Dict]:
        """Get all readings for a specific device"""
        return [
            reading for reading in self.sensor_readings 
            if reading["device_id"] == device_id
        ]
    
    def get_recent_readings(self, hours: int = 24) -> List[Dict]:
        """Get readings from the last N hours"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_readings = []
        
        for reading in self.sensor_readings:
            reading_time = datetime.fromisoformat(reading["timestamp"])
            if reading_time >= cutoff_time:
                recent_readings.append(reading)
        
        return recent_readings
    
    def update_device_status(self, device_id: str, new_status: str):
        """Update device status"""
        for device in self.devices:
            if device["id"] == device_id:
                old_status = device["status"]
                device["status"] = new_status
                
                # Log maintenance activity
                if new_status == "maintenance":
                    self.maintenance_log.append({
                        "device_id": device_id,
                        "action": "maintenance_started",
                        "timestamp": datetime.now().isoformat()
                    })
                
                print(f"📱 Device {device_id} status: {old_status} → {new_status}")
                break
    
    def remove_old_readings(self, days: int = 30):
        """Remove readings older than specified days"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(days=days)
        initial_count = len(self.sensor_readings)
        
        self.sensor_readings = [
            reading for reading in self.sensor_readings
            if datetime.fromisoformat(reading["timestamp"]) >= cutoff_time
        ]
        
        removed_count = initial_count - len(self.sensor_readings)
        print(f"🗑️  Removed {removed_count} old readings")
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        online_devices = [d for d in self.devices if d["status"] == "online"]
        offline_devices = [d for d in self.devices if d["status"] == "offline"]
        maintenance_devices = [d for d in self.devices if d["status"] == "maintenance"]
        
        return {
            "total_devices": len(self.devices),
            "online_devices": len(online_devices),
            "offline_devices": len(offline_devices),
            "maintenance_devices": len(maintenance_devices),
            "total_readings": len(self.sensor_readings),
            "active_alerts": len(self.alerts),
            "maintenance_activities": len(self.maintenance_log)
        }
    
    def export_data(self) -> Dict:
        """Export all data for backup"""
        return {
            "devices": self.devices,
            "sensor_readings": self.sensor_readings,
            "alerts": self.alerts,
            "maintenance_log": self.maintenance_log,
            "export_timestamp": datetime.now().isoformat()
        }

# Usage Example
if __name__ == "__main__":
    # Create IoT data manager
    iot_manager = IoTDataManager()
    
    # Register devices
    iot_manager.register_device("TEMP_001", "temperature", "server_room")
    iot_manager.register_device("HUM_002", "humidity", "server_room")
    iot_manager.register_device("PRESS_003", "pressure", "lab")
    
    # Update device statuses
    iot_manager.update_device_status("TEMP_001", "online")
    iot_manager.update_device_status("HUM_002", "online")
    iot_manager.update_device_status("PRESS_003", "maintenance")
    
    # Add sensor readings
    iot_manager.add_sensor_reading("TEMP_001", 25.5, "°C")
    iot_manager.add_sensor_reading("TEMP_001", 32.1, "°C")  # This will trigger alert
    iot_manager.add_sensor_reading("HUM_002", 65.2, "%")
    
    # Get system status
    status = iot_manager.get_system_status()
    print("\n📊 System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Get device readings
    temp_readings = iot_manager.get_device_readings("TEMP_001")
    print(f"\n🌡️  Temperature readings: {len(temp_readings)} records")
```

