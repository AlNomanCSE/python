# Python Printing Methods Guide 🐍


A comprehensive guide to all the different ways you can print variables and data in Python. Perfect for beginners learning Python or developers looking for the best printing practices.

## 🚀 Quick Start

```python
# Sample IoT sensor data
temperature = 23.7      
humidity = 65            
device_name = "TempSensor01"  
is_online = True        

# Modern way (recommended)
print(f"Temperature: {temperature}°C")
print(f"Humidity: {humidity}%")
print(f"Device: {device_name} is {'online' if is_online else 'offline'}")
```

## 📋 Table of Contents

- [Printing Methods](#printing-methods)
  - [Basic Methods](#basic-methods)
  - [String Formatting](#string-formatting)
  - [Advanced Formatting](#advanced-formatting)
  - [Specialized Output](#specialized-output)
- [Best Practices](#best-practices)
- [Use Cases](#use-cases)
- [Examples](#examples)


## 📝 Printing Methods

### Basic Methods

#### 1. Simple Print Statements
```python
print(temperature)        # 23.7
print(humidity)          # 65
print(device_name)       # TempSensor01
print(is_online)         # True
```

#### 2. Multiple Values
```python
# Print multiple values
print(temperature, humidity, device_name, is_online)
# Output: 23.7 65 TempSensor01 True

# With custom separator
print(temperature, humidity, device_name, is_online, sep=" | ")
# Output: 23.7 | 65 | TempSensor01 | True
```

#### 3. String Concatenation
```python
print("Temperature: " + str(temperature))
print("Device: " + device_name)
```

### String Formatting

#### 4. Format Method (.format())
```python
print("Temperature: {}°C".format(temperature))
print("Humidity: {}%".format(humidity))
print("Temp: {:.1f}°C, Humidity: {}%".format(temperature, humidity))
```

#### 5. F-Strings (⭐ Recommended)
```python
print(f"Temperature: {temperature}°C")
print(f"Humidity: {humidity}%")
print(f"Device: {device_name}")
print(f"Status: {'Online' if is_online else 'Offline'}")
```

#### 6. Printf-Style Formatting
```python
print("Temperature: %.1f°C" % temperature)
print("Humidity: %d%%" % humidity)
```

### Advanced Formatting

#### 7. Aligned Output
```python
print(f"{'Parameter':<15} {'Value':<15} {'Unit':<10}")
print("-" * 40)
print(f"{'Temperature':<15} {temperature:<15.1f} {'°C':<10}")
print(f"{'Humidity':<15} {humidity:<15} {'%':<10}")
```

Output:
```
Parameter       Value           Unit      
----------------------------------------
Temperature     23.7            °C        
Humidity        65              %         
```

#### 8. Multiline Strings
```python
sensor_report = f"""
IoT Sensor Report
-----------------
Temperature: {temperature}°C
Humidity: {humidity}%
Device: {device_name}
Status: {'Online' if is_online else 'Offline'}
"""
print(sensor_report)
```

#### 9. Dictionary Formatting
```python
sensor_data = {
    'temp': temperature,
    'humidity': humidity,
    'device': device_name,
    'online': is_online
}
print("Temperature: {temp}°C, Device: {device}".format(**sensor_data))
```

### Specialized Output

#### 10. File Output
```python
with open("sensor_log.txt", "w") as file:
    print(f"Temperature: {temperature}°C", file=file)
    print(f"Humidity: {humidity}%", file=file)
```

#### 11. Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Temperature: {temperature}°C")
logging.info(f"Device {device_name} status: {'online' if is_online else 'offline'}")
```

#### 12. JSON Format
```python
import json
sensor_json = {
    "temperature": temperature,
    "humidity": humidity,
    "device_name": device_name,
    "is_online": is_online
}
print(json.dumps(sensor_json, indent=2))
```

#### 13. Pretty Printing
```python
from pprint import pprint
complex_data = {
    "sensors": {
        "temperature": {"value": temperature, "unit": "°C"},
        "humidity": {"value": humidity, "unit": "%"}
    },
    "device": {"name": device_name, "online": is_online}
}
pprint(complex_data)
```

## 🎯 Best Practices

### ✅ Recommended Approaches

| Method | Use Case | Example |
|--------|----------|---------|
| **F-strings** | General purpose, most readable | `f"Temp: {temp}°C"` |
| **Format method** | Complex formatting needs | `"{:.2f}".format(value)` |
| **Logging** | Production applications | `logging.info(f"Status: {status}")` |
| **JSON** | Data exchange, APIs | `json.dumps(data, indent=2)` |

### ⚡ Performance Comparison

```python
# Fastest to slowest:
# 1. F-strings (fastest)
f"Temperature: {temperature}°C"

# 2. Format method
"Temperature: {}°C".format(temperature)

# 3. Concatenation (slowest)
"Temperature: " + str(temperature) + "°C"
```

### 🎨 Styling Tips

```python
# Use emojis for visual appeal
print(f"🌡️  Temperature: {temperature}°C")
print(f"💧 Humidity: {humidity}%")
print(f"{'🟢 ONLINE' if is_online else '🔴 OFFLINE'}")

# Create borders and boxes
print("┌─ Sensor Data ─┐")
print(f"│ Temp: {temperature:>6.1f}°C │")
print(f"│ Humid: {humidity:>5}%   │")
print("└───────────────┘")
```

## 💡 Use Cases

### IoT Applications
```python
def print_sensor_status(temp, humidity, device, online):
    status = "🟢 ONLINE" if online else "🔴 OFFLINE"
    print(f"Device: {device}")
    print(f"Temperature: {temp:.1f}°C")
    print(f"Humidity: {humidity}%")
    print(f"Status: {status}")
```

### Data Logging
```python
import datetime

def log_sensor_data(temp, humidity, device):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {device}: Temp={temp}°C, Humidity={humidity}%"
    
    # Print to console
    print(log_entry)
    
    # Save to file
    with open("sensor.log", "a") as f:
        print(log_entry, file=f)
```

### Debug Information
```python
def debug_print(variable_name, value):
    print(f"DEBUG: {variable_name} = {value} (type: {type(value).__name__})")

debug_print("temperature", temperature)
debug_print("is_online", is_online)
```

## 📊 Examples

### Complete IoT Dashboard
```python
def display_dashboard(sensors_data):
    print("=" * 50)
    print("🏠 SMART HOME DASHBOARD")
    print("=" * 50)
    
    for sensor in sensors_data:
        status_icon = "🟢" if sensor['online'] else "🔴"
        print(f"{status_icon} {sensor['name']:<15} | "
              f"Temp: {sensor['temp']:>6.1f}°C | "
              f"Humidity: {sensor['humidity']:>3}%")
    
    print("=" * 50)

# Usage
sensors = [
    {"name": "Living Room", "temp": 23.7, "humidity": 65, "online": True},
    {"name": "Bedroom", "temp": 22.1, "humidity": 58, "online": True},
    {"name": "Kitchen", "temp": 25.3, "humidity": 72, "online": False}
]

display_dashboard(sensors)
```

### Error Handling with Print
```python
def safe_print_sensor(temp, humidity, device):
    try:
        if not isinstance(temp, (int, float)):
            raise ValueError("Temperature must be a number")
        
        print(f"✅ {device}: {temp:.1f}°C, {humidity}%")
        
    except ValueError as e:
        print(f"❌ Error printing {device} data: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
```

