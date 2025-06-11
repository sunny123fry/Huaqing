# Python与Git基础课程学习笔记

## 学习内容1：变量、变量类型、作用域
```python
# 变量类型
name = "Alice"  # str
age = 20        # int
grades = [90, 85, 88]  # list
info = {"name": "Alice", "age": 20}  # dict

# 类型转换
age_str = str(age)
number = int("123")

# 作用域
x = 10  # 全局变量
def my_function():
    y = 5  # 局部变量
    global x
    x += 1
    print(f"Inside function: x={x}, y={y}")

my_function()
print(f"Outside function: x={x}")
```

### 学习心得
- **动态类型**：变量类型由赋值决定，无需显式声明
- **作用域规则**：局部变量 > 全局变量
- **`global`关键字**：用于在函数内修改全局变量

#### 常见错误与解决方案
```python
x = 10
def func():
    # 错误：未声明global直接修改全局变量
    x += 1  # UnboundLocalError
    
    # 正确做法
    global x
    x += 1
```

---

## 学习内容2：运算符及表达式
```python
# 算术运算
a = 10
b = 3
print(a + b)  # 13
print(a // b)  # 3（整除）
print(a ** b)  # 1000（幂）

# 逻辑运算
x = True
y = False
print(x and y)  # False
print(x or y)   # True

# 比较运算
print(a > b)  # True
```

### 学习心得
- **算术运算**：
  - `//` 整除 vs `/` 真除
  - `**` 幂运算优先级高于乘除
- **逻辑运算**：
  - 短路求值：`and`/`or`在确定结果后停止计算
- **浮点数精度**：
  ```python
  print(0.1 + 0.2 == 0.3)  # False (实际≈0.30000000000000004)
  ```

---

## 学习内容3：语句（条件、循环、异常）
```python
# 条件语句
score = 85
if score >= 90:
    print("A")
elif score >= 60:
    print("Pass")
else:
    print("Fail")

# 循环语句
for i in range(5):
    if i == 3:
        continue
    print(i)

# 异常处理
try:
    num = int(input("Enter a number: "))
    print(100 / num)
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid input!")
finally:
    print("Execution completed.")
```

### 学习心得
- **条件结构**：`if-elif-else`注意条件顺序
- **循环控制**：
  - `break`：完全退出循环
  - `continue`：跳过当前迭代
- **异常处理金字塔**：
  ```python
  try:
      # 可能出错的代码
  except SpecificError:  # 优先捕获具体异常
  except Exception:      # 通用异常兜底
  finally:               # 清理资源
  ```

#### 常见错误
```python
# 避免裸except吞掉所有异常
try:
    risky_call()
except:  # 危险！应指定异常类型
    pass
```

---

## 学习内容4：函数
```python
# 函数定义
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))  # Hello, Alice!
print(greet("Bob", "Hi"))  # Hi, Bob!

# 可变参数
def sum_numbers(*args):
    return sum(args)
print(sum_numbers(1, 2, 3, 4))  # 10

# 匿名函数
double = lambda x: x * 2
print(double(5))  # 10

# 高阶函数
def apply_func(func, value):
    return func(value)
print(apply_func(lambda x: x ** 2, 4))  # 16
```

### 学习心得
- **参数传递**：
  ```python
  def func(a, *args, b=1, **kwargs):
      # a: 位置参数
      # args: 可变位置参数(元组)
      # b: 关键字参数
      # kwargs: 可变关键字参数(字典)
  ```
- **默认值陷阱**：
  ```python
  # 错误：可变默认值会累积
  def add_item(item, lst=[]):
      lst.append(item)
      return lst
      
  # 正确：使用None作为哨兵值
  def add_item(item, lst=None):
      lst = lst or []
      lst.append(item)
      return lst
  ```

---

## 学习内容5：包和模块
```python
# 创建模块 mymodule.py
# mymodule.py
def say_hello():
    return "Hello from module!"

# 主程序
import mymodule
print(mymodule.say_hello())

# 导入第三方模块
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200

# 包使用示例
from mypackage import mymodule
```

### 学习心得
1. **标准包结构**：
   ```
   mypackage/
       __init__.py
       module1.py
       subpackage/
           __init__.py
           module2.py
   ```
2. **导入方式**：
   ```python
   import module               # 标准导入
   from . import submodule     # 相对导入
   import pandas as pd         # 别名导入
   ```
3. **依赖管理**：
   ```bash
   pip install requests
   pip freeze > requirements.txt
   ```

#### 常见问题
```python
# 循环导入解决方案：
# 1. 重构代码消除依赖
# 2. 在函数内部导入模块
```

---

## 学习内容6：类和对象
```python
# 定义类
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I am {self.name}, {self.age} years old."

# 继承
class GradStudent(Student):
    def __init__(self, name, age, major):
        super().__init__(name, age)
        self.major = major

    def introduce(self):
        return f"I am {self.name}, a {self.major} student."

# 使用
student = Student("Alice", 20)
grad = GradStudent("Bob", 22, "CS")
print(student.introduce())  # I am Alice, 20 years old.
print(grad.introduce())     # I am Bob, a CS student.
```

### 学习心得
- **类变量 vs 实例变量**：
  ```python
  class MyClass:
      class_var = []  # 所有实例共享
      
      def __init__(self):
          self.instance_var = []  # 实例独有
  ```
- **继承**：使用`super()`调用父类方法
- **多态**：子类重写父类方法实现不同行为

---

## 学习内容7：装饰器
```python
# 简单装饰器
def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

# 带参数的装饰器
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hi, {name}!")

greet("Alice")
```

### 学习心得
- **装饰器本质**：
  ```python
  @decorator
  def func(): ...
  
  # 等价于：
  func = decorator(func)
  ```
- **带参数装饰器**（三层嵌套）：
  ```python
  def repeat(n):
      def decorator(func):
          def wrapper(*args):
              for _ in range(n):
                  func(*args)
          return wrapper
      return decorator
  ```
- **保留元数据**：
  ```python
  from functools import wraps
  
  def my_decorator(func):
      @wraps(func)  # 保留原函数信息
      def wrapper(*args, **kwargs):
          return func(*args, **kwargs)
      return wrapper
  ```

---

## 学习内容8：文件操作
```python
# 写文件
with open("example.txt", "w") as f:
    f.write("Hello, Python!\n")

# 读文件
with open("example.txt", "r") as f:
    content = f.read()
    print(content)

# 处理CSV
import csv
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 20])
```

### 学习心得
- **资源管理**：始终使用`with`语句自动关闭文件
- **文件模式**：
  | 模式 | 说明                  |
  |------|-----------------------|
  | `r`  | 读取（默认）          |
  | `w`  | 写入（覆盖）          |
  | `a`  | 追加                  |
  | `b`  | 二进制模式            |
- **CSV处理**：使用`csv`模块处理特殊字符

#### 经典错误
```python
# 错误：忘记关闭文件（数据可能丢失）
f = open("data.txt", "w")
f.write("important data")
# 崩溃时数据可能未写入磁盘！

# 正确：使用with语句保证关闭
with open("data.txt", "w") as f:
    f.write("guaranteed write")
```

---

## 综合编程建议
1. **作用域管理**：避免过度使用`global`，优先返回值传递数据
2. **异常处理**：具体异常 > 通用异常 > 裸except
3. **OOP原则**：组合 > 继承（避免深度继承链）
4. **模块设计**：单一职责原则（模块代码<500行）
5. **调试技巧**：使用`__name__`隔离测试代码
   ```python
   if __name__ == "__main__":
       # 仅在该模块直接运行时执行
       test_function()
   ```