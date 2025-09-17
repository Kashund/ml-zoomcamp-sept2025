
-----

## 📋 Git & GitHub Crash Course Cheatsheet

This guide provides a quick reference for the most common commands for your terminal, Git, and GitHub.

### **Basic Terminal Command**

Before using Git, you'll need to navigate your computer using the command line. The `ls` command is one of the most fundamental.

  * **`ls` (List)**: Lists the files and directories in your current location.
      * **Example:** `ls`
        > `Desktop Documents Downloads Music`
      * **Example (List all, including hidden files):** The `-a` flag shows hidden files, which is useful for seeing the `.git` directory.
        `ls -a`
        > `. .. .git README.md index.html`
      * **Tip:** Think of `ls` as opening a folder in a file explorer to see what's inside.

-----

### **Initial Setup**

This only needs to be done once. This information is attached to every commit you make.

  * **`git config --global user.name "Your Name"`**: Sets the name that will be attached to your commits.
      * **Example:** `git config --global user.name "Ada Lovelace"`
      * **Tip:** Use the name you want to be displayed on GitHub.
  * **`git config --global user.email "youremail@example.com"`**: Sets the email that will be attached to your commits.
      * **Example:** `git config --global user.email "ada@example.com"`
      * **Tip:** Use the same email address you used to sign up for GitHub.
  * **`git config --list`**: Checks your current configuration settings.

-----

### **Creating & Cloning Repositories**

Start a new project or get a copy of an existing one.

  * **`git init`**: Initializes a new Git repository in your current folder.
      * **Tip:** Run this command inside the main folder of your new project. You'll see a message like "Initialized empty Git repository in /path/to/your/project/.git/".
  * **`git clone <repository_url>`**: Creates a local copy of a repository from a URL (like one from GitHub).
      * **Example:** `git clone https://github.com/torvalds/linux.git`
      * **Tip:** This is the most common way to start working on a project that already exists. It automatically sets up the connection to the remote repository for you.

-----

### **The Basic Workflow 🔄**

The core process for saving your changes.

  * **`git status`**: Checks the status of your files (untracked, modified, or staged).
      * **Tip:** Run this command often\! It's your "are we there yet?" for understanding what's going on in your repository.
  * **`git add <file_name>`**: Stages a specific file for your next commit. Staging is like putting an item in a shopping cart before you check out.
      * **Example:** `git add README.md`
  * **`git add .`**: Stages all modified and new files in the current directory.
      * **Tip:** This is a convenient shortcut, but use `git status` first to make sure you're not accidentally adding files you want to ignore.
  * **`git commit -m "Your descriptive message"`**: Commits your staged changes, creating a snapshot in the project's history.
      * **Example:** `git commit -m "feat: Add user login functionality"`
      * **Tip:** Write clear, concise commit messages. The convention is to use the present tense (e.g., "Add feature" not "Added feature").
  * **`git push`**: Pushes your committed changes from your local machine to the remote repository (e.g., GitHub).
      * **Example:** `git push origin main`
      * **Tip:** You must commit your changes before you can push them.

-----

### **Branching & Merging 🌿**

Work on different features in isolation without affecting the main codebase.

  * **`git branch`**: Lists all local branches. The current one is marked with an asterisk `*`.
  * **`git branch <new_branch_name>`**: Creates a new branch.
      * **Example:** `git branch feature/user-profile`
  * **`git checkout <branch_name>`**: Switches to a different branch.
      * **Example:** `git checkout feature/user-profile`
      * **Tip:** You can combine creating and switching to a new branch with one command: `git checkout -b <new_branch_name>`.
  * **`git merge <branch_name_to_merge>`**: Merges the specified branch's history into your current branch.
      * **Example:** First, switch to the receiving branch: `git checkout main`. Then, merge the other branch into it: `git merge feature/user-profile`.
  * **`git branch -d <branch_name>`**: Deletes a branch.
      * **Example:** `git branch -d feature/user-profile`
      * **Tip:** You can only delete a branch after its changes have been successfully merged.

-----

### **Working with Remotes ☁️**

Manage your connections to remote repositories.

  * **`git remote -v`**: Lists your remote repositories with their URLs. `origin` is the default name for the remote you cloned from.
  * **`git fetch <remote_name>`**: Downloads changes from a remote repository but doesn't integrate them into your working files.
      * **Tip:** This is useful for seeing what others have been working on without automatically merging it into your own work.
  * **`git pull <remote_name> <branch_name>`**: Fetches changes from a remote and automatically merges them into your current branch.
      * **Example:** `git pull origin main`
      * **Tip:** It's good practice to `pull` the latest changes from the main branch before starting new work to avoid conflicts.

-----

### **Other Useful Commands & Concepts**

  * **`git log`**: Views the detailed commit history. Press `q` to exit.
  * **`git log --oneline --graph --all`**: Views a condensed, graphical log of all branches.
      * **Tip:** This is one of the most useful log commands for getting a quick overview of the entire project history.
  * **`git diff`**: Shows the exact changes (additions and deletions) between your working directory and the staging area.
  * **`git stash`**: Temporarily saves uncommitted changes so you can switch branches.
      * **Tip:** This is a lifesaver when you're in the middle of a feature and need to quickly switch branches to fix a bug. After, run `git stash pop` to get your changes back.
  * **`.gitignore` File**: A text file where you list files and folders that Git should ignore (like log files or secret keys).
      * **Example `.gitignore` content:**
        ```
        # Dependencies
        node_modules/

        # Environment variables
        .env
        ```
      * **Tip:** Every project should have a `.gitignore` file. You can find standard templates online for different languages and frameworks.

-----

### **Undoing Changes ⏪**

Fixing mistakes and reverting to previous states.

  * **`git reset <file_name>`**: Unstages a file, removing it from the staging area but keeping the changes.
      * **Example:** `git reset README.md`
  * **`git commit --amend -m "New message"`**: Amends the most recent commit. This is useful for fixing a typo in a commit message or adding a file you forgot.
      * **Example:** `git add forgotten_file.js`, followed by `git commit --amend --no-edit` to add the file without changing the message.
      * **Tip:** Avoid amending commits that have already been pushed to a shared branch, as it rewrites history.
  * **`git checkout -- .`**: Discards all uncommitted changes in your working directory. **⚠️ This is permanent and cannot be undone.**
      * **Tip:** Be absolutely sure you want to throw away your work before running this command.
  * **`git revert <commit_hash>`**: Creates a new commit that undoes the changes from a specified commit.
      * **Example:** Find a commit hash using `git log`, then run `git revert a1b2c3d4`.
      * **Tip:** This is the safest way to undo changes on a shared/public branch because it doesn't rewrite history.