import curses

def interactive_mode(stdscr):
    a = 0
    b = 0

    curses.noecho()
    curses.cbreak()
    stdscr.keypad(1)

    stdscr.addstr(0, 10, "Press 'a', 'b' or 'q'...")
    stdscr.refresh()

    while True:
        key = stdscr.getch()
        if key == ord('a'):
            a += 1
            stdscr.addstr(1, 10, f"Variable 'a' incremented. Value: {a}      ")  # Added spaces to clear previous message
            stdscr.refresh()
        elif key == ord('b'):
            b += 1
            stdscr.addstr(1, 10, f"Variable 'b' incremented. Value: {b}      ")
            stdscr.refresh()
        elif key == ord('q'):
            stdscr.addstr(1, 10, "Exiting interactive mode.              ")
            stdscr.refresh()
            break
        else:
            stdscr.addstr(1, 10, f"Unknown key: {chr(key)}. Press 'a', 'b' or 'q'...      ")
            stdscr.refresh()

# Run the function
curses.wrapper(interactive_mode)
