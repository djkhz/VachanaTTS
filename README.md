# VahanaTTS (VITS Models)

## การใช้งาน (ภาษาไทย)

VahanaTTS เป็นระบบ Text-to-Speech (TTS) ที่ใช้โมเดล VITS ซึ่งช่วยให้คุณสามารถแปลงข้อความเป็นเสียงได้อย่างมีประสิทธิภาพ รองรับการใช้งานทั้งบน GPU และ CPU

---

## 1. การติดตั้ง

### ขั้นตอนการติดตั้ง
1. เปิด Command Prompt หรือ Terminal
2. ใช้คำสั่งต่อไปนี้:

```sh
git clone https://github.com/VYNCX/VachanaTTS.git
cd VahanaTTS
```

3. รันไฟล์ `install.bat` เพื่อติดตั้งโปรแกรม:
   - หากต้องการใช้งาน GPU ให้พิมพ์ `y` (ต้องใช้ NVIDIA CUDA)
     - **ข้อดี**: รวดเร็ว
     - **ข้อเสีย**: ใช้ทรัพยากรเครื่องและพื้นที่เยอะ
   - หากต้องการใช้งาน CPU ให้พิมพ์ `n`
     - **ข้อดี**: ประหยัดพื้นที่และทรัพยากร
     - **ข้อเสีย**: ช้ากว่า โดยเฉพาะหากใช้งานโหมดโคลนเสียง

---

## 2. การใช้งาน

### เริ่มต้นใช้งาน
1. รันคำสั่งต่อไปนี้ใน Command Prompt หรือ Terminal:

```sh
python app.py
```

หรือ

2. ใช้ไฟล์ `run.bat` เพื่อเริ่มโปรแกรม

---

## 3. ดาวน์โหลดโมเดล

ก่อนใช้งาน จำเป็นต้องดาวน์โหลดโมเดลจาก [Huggingface](https://huggingface.co) ตัวอย่างโมเดล:

[VIZINTZOR - Huggingface](https://huggingface.co/VIZINTZOR)

1. ดาวน์โหลดโมเดลที่ต้องการ
2. วางไฟล์โมเดลในโฟลเดอร์ `models` ของโปรเจกต์

---

## 4. Finetune โมเดล ด้วย dataset ของตัวเอง

[Finetune Colab](https://colab.research.google.com/drive/12qbpHnu7wYiTEoqh6_57_KUjp4gJkx2h?usp=sharing)

