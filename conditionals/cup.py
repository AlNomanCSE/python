# cup = input("Choose your cup size (small/medium/large) : ").lower();
# if cup=="small" :
#     print("Price is 10 rupees")
# elif cup=="medium":
#     print("Price is 20 rupees")
# else:
#     print("Price is 30 rupees")



status_code = 404

match status_code:
    case 200:
        message = "Success!"
    case 404:
        message = "Page Not Found." # This block executes
    case 500 | 503: # Multi-case support
        message = "Server Error."
    case _: # The catch-all (like the 'default' case)
        message = "Unknown Status."

print(message)
# Output: Page Not Found.
