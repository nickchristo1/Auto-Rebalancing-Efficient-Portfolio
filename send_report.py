# Nicholas Christophides  nick.christophides@gmail.com

import smtplib
from email.message import EmailMessage
import matplotlib.pyplot as plt
import io
import os
from tabulate import tabulate
from backtest import table_data
from auto_rebalance import buy_orders, sell_orders
from datetime import date


# 1.) Generate image of position adjustments
all_orders = buy_orders | sell_orders
print(all_orders)
filtered_orders = {k: v for k, v in all_orders.items() if v != 0}  # Leave out non-adjusted positions

keys = list(filtered_orders.keys())
values = list(filtered_orders.values())

plt.clf()
plt.bar(keys, values)
plt.xlabel('Ticker')
plt.ylabel('Order Value')
plt.title('Position Adjustments')
plt.xticks(rotation=45)
plt.tight_layout()

# 2.) Save image and prepare for entry in report
image_buffer = io.BytesIO()
plt.savefig(image_buffer, format='png')  # Save image to buffer
plt.close()


# 3.) Function to send email of report

def send_email(sender_email, sender_password, recipient_email, body_text, image_bytes):
    msg = EmailMessage()
    msg['Subject'] = f"Auto-Rebalance Report on {date.today()}"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    msg.set_content(body_text)

    msg.add_attachment(
        image_bytes,
        maintype='image',
        subtype='png',
        filename='position_adjustments.png'
    )

    try:
        print("Connecting to server...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")


# 4.) Gather information to send as email
email = "nick.christophides@gmail.com"
password = os.getenv('GMAIL_PASSCODE')
table_string = tabulate(table_data,
                        headers=["Metric", "Strategy", "SPY Buy & Hold"],
                        tablefmt="grid",
                        stralign="right")

send_email(
    sender_email=email,
    sender_password=password,
    recipient_email=email,
    body_text=f"Good Morning,\nThe rebalancing for this week has been executed. A summary table of statistics and a"
              f" figure detailing the orders are attached.\n\n{table_string}",
    image_bytes=image_buffer.getvalue()
)
