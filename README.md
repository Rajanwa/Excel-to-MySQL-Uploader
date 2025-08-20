# 📊 MySQL Excel Data Uploader Through Python with PDF Report Generator

A powerful Python desktop application to upload Excel files directly into MySQL databases with authentication and generate PDF reports summarizing uploaded records. Built for data analysts, finance teams, and backend admins.

---

## 🚀 Features

- 🔐 **Secure Authentication** with Host, Username, and Password
- 📁 **Upload Excel File** directly into a MySQL database table
- 🗃️ Supports dynamic selection of `database.table`
- 📄 **Generate PDF Report** summarizing uploaded data
- 📦 Real-time **Upload Status Console**
- 🧠 Built with a clean & responsive GUI using `tkinter`
- ✅ Handles both `.xls` and `.xlsx` formats

---

## 🖥️ GUI Overview

| Section             | Description |
|---------------------|-------------|
| **Database Connection** | Input host, username, password, and select target database.table |
| **File Upload**      | Choose and upload an Excel file |
| **Generate PDF**     | Click to generate a report of all inserted data |
| **Upload Status**    | Console shows real-time success/error messages |

---

## 🛠️ Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/mysql-excel-uploader.git
cd mysql-excel-uploader
