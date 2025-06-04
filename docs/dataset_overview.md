# 📄 Dataset Overview – Credit Scoring Project

This dataset combines customer profile data, credit bureau history, and loan application details to predict whether a customer will default on a loan (`default = 1`) or not (`default = 0`).

## 🎯 Target Variable

| Column Name | Description |
|-------------|-------------|
| **default** | Binary indicator (0 = no default, 1 = default). This is the target variable we aim to predict – whether the customer will fail to repay the loan. |

---

## 🧾 Loan Application Data

| Column Name | Description |
|-------------|-------------|
| **loan_id** | Unique identifier for each loan application. |
| **cust_id** | Customer ID – links customer data across tables. |
| **loan_purpose** | Reason for the loan (e.g., Personal, Auto, Home, etc.). |
| **loan_type** | Type of loan: typically `Secured` or `Unsecured`. |
| **sanction_amount** | Total loan amount initially approved by the bank. |
| **loan_amount** | Actual loan amount disbursed to the customer. |
| **processing_fee** | Fee charged by the bank for processing the loan. |
| **gst** | Goods and Services Tax applied on the processing fee. |
| **net_disbursement** | Final amount credited to the customer = loan_amount - fees - taxes. |
| **loan_tenure_months** | Duration of the loan in months. |
| **principal_outstanding** | Remaining unpaid principal amount at the time of observation. |
| **bank_balance_at_application** | Bank balance of the customer when applying for the loan. |
| **disbursal_date** | Date on which the loan was disbursed. |
| **installment_start_dt** | Date on which EMI (installments) started. |

---

## 🧮 Credit Bureau History (Behavioral Data)

| Column Name | Description |
|-------------|-------------|
| **number_of_open_accounts** | Number of currently active loan or credit card accounts. |
| **number_of_closed_accounts** | Number of fully paid/closed loan or credit card accounts. |
| **total_loan_months** | Total number of months across all loan exposures. |
| **delinquent_months** | Total number of months with payment delays. |
| **total_dpd** | Days Past Due – cumulative days a customer was late with payments. |
| **enquiry_count** | Number of recent loan/credit inquiries made by the customer. |
| **credit_utilization_ratio** | Ratio of credit currently used to total available credit – proxy for financial pressure. |

---

## 👤 Customer Demographics & Profile

| Column Name | Description |
|-------------|-------------|
| **age** | Age of the customer (in years). |
| **gender** | Gender of the customer (`Male`, `Female`, etc.). |
| **marital_status** | Marital status (`Single`, `Married`, etc.). |
| **employment_status** | Employment status (`Salaried`, `Self-employed`, etc.). |
| **income** | Monthly or annual income of the customer (numerical). |
| **number_of_dependants** | Number of people financially dependent on the customer. |
| **residence_type** | Type of housing (`Owned`, `Rented`, `Company Provided`, etc.). |
| **years_at_current_address** | How long the customer has lived at the current address (in years). |
| **city** | City of residence. |
| **state** | State or province of residence. |
| **zipcode** | Postal code of the residence. |

