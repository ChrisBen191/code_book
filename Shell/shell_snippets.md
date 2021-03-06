
# SHELL
---
# NAVIGATION COMMANDS
|      Commands       |                                                                                                 |
| :-----------------: | ----------------------------------------------------------------------------------------------- |
|        `pwd`        | Displays the **present working directory**, or the path to the current directory                |
|        `ls`         | **Lists** all non-hidden items/documents in the current directory; add '-A' flag to show hidden |
|  `cd folder_name`   | Used to **change directory**, or to navigate into the directory/folder specified                |
|      `history`      | Provides the **historical** commands passed through the Shell                                   |
| `man shell_command` | Provides the **manual** for the shell command specified                                         |
|  ` cat file-name`   | Displays the first few lines of a file in the shell                                             |

# MODIFIER COMMANDS

|          Commands           |                                                                                  |
| :-------------------------: | -------------------------------------------------------------------------------- |
| `touch file_name.file_type` | **Creates a file** in the current location/directory, in the file type specified |
|       `rm file_name`        | **Deletes the file** specified (cannot delete a folder/directory)                |
|     `mkdir folder_name`     | **Creates a new directory** in the current directory                             |
|    `rm -rf folder_name`     | **Deletes an entire folder/directory** and its contents                          |

Displays or prints the variable or string specified
```console
echo hello world!
```

**Moves** the specified **file** to the directory specified
```console
mv file_to_be_moved    directory/path/to/move/file
```

**Copies** the specified **file** to the specified directory
```console
cp file_to_be_moved    directory/path/to/copy/file
```

Moves the specified folder to the specified directory
```console
mv -R folder_to_be_moved    directory/path/to/move/folder
```

**Copies** the specified **folder** to the specified directory
```console
cp -R folder_to_be_moved    directory/path/to/copy/folder
```

Searches and displays the **term** specified in the **file(s)** specified
```console
grep term file

# FLAGS
-c : count of matching lines, not matching text
-h : do NOT print names of files when searching multiple files
-i : ignore case sensitivity
-l : print names of matching files, not matching text
-n : print line numbers for matching lines
-v : invert the match; show lines that DON'T match
```
Counts and displays the number of character, words, or lines in a file specified
```console
wc file_name

# FLAGS
-c : counts characters in file
-w : counts words in file
-l : counts lines in file
```

# GITHUB
---
## MODIFIER COMMANDS

Initializes a repository to be tracked by Git (must be inside of folder you wish to track)
```console
git init
```
Clones the specified respository to the local machine
```console
git clone "URL-OF-REPOSITORY-TO-BE-CLONED"
```

Shows the status of the current local repository; will show up-to-date, behind master, etc.
```console
git status
```

Stages all unstaged files to be uploaded
```console
git add -A
```

Removes a file added in an earlier commit
```console
git rm <filename>
```

Commits the staged files to be pushed to remote repository; requires message
```console
git commit -m "put your message here(usually what was added)"	
```

Pushes the committed files to remote repository
```console
git push	
```

Resets the local machine to last instance pulled from master. **All local changes will be lost.**
```console
git reset --hard
```

Stashes local changes to the repository before performing a 'git pull'. **Local changes will not be lost.**
```console
git stash	
```

Updates local machine from a remote repository
```console        
git pull
```
Performed after a **git pull**; reapplies stashed local files and changes to the repository
```console
git stash pop	
```

# CONDA 
---

|            Commands             |                                                                |
| :-----------------------------: | -------------------------------------------------------------- |
|        `conda env list`         | displays a list of all **environments** on the system          |
|          `conda list`           | displays all packages installed in **the current environment** |
|   `conda create -n env-name`    | **creates** the environment with the name specified            |
| `conda env remove --n env-name` | **deletes** the environment specified from the system          |
|    `conda activate env-name`    | activates the **environment** specified                        |
|       `conda deactivate`        | **deactivates** the current environment in use                 |

creates a **yml file** that contains the exact versions of packages in an environment for reproducibility
```console
conda env export > env-name.yml

# FLAGS
-n : specify the name of environment to export if not current active environment
-f : saves the environment info to a file; can use 'shell piping' instead
```

imports a **yml file** to reproduce an environment with exact versions of the packages being used 
```console
conda env create -f env-name.yml
```

converts the specified jupyter notebook into the format specified (typically '--to script')
```console
jupyter nbconvert --to FORMAT notebook.ipynb

# FLAG
-to : script, markdown, pdf, latex, asciidoc
```

# LINUX
display which version of the Linux kernel is running
```console
uname -srm

# provide a longer, more descriptive version
uname --kernel-name --kernel-release --machine
```

display information on what distribution is running on a system
```console
cat /etc/os-release`
```

creates a **log of the output of a .py file** to the file specified
```console
ipython python_file.py prod 2>&1 | tee log/output_file.txt

-- 'ipython' or 'python' can be used
-- 'prod 2>&1' indicates to record "standard errors" (2) in the same location as "standard output" (1)
-- 'tee' displays output and records it to log/output_file.txt 
```
## TMUX
| Commands `C-b==CTRL+b` `C-d==CTRL+d` |                                                                 |
| :----------------------------------: | --------------------------------------------------------------- |
|      `tmux new -s session-name`      | creates new session named `session-name`                        |
|              `tmux ls`               | displays all **sessions** currently running (detached/attached) |
|          `C-b d` or `C-b D`          | detaches current **session** *(D allows selection)*             |
| `tmux a -t session-name` OR `tmux a` | attaches to `session-name` OR will connect to last session      |
|        `tmux kill-session -a`        | kill all sessions but the **current session**                   |
| `tmux kill-session -t session-name`  | kill `session-name` specified                                   |
|               `C-b "`                | splits **panes** horizontally                                   |
|               `C-b %`                | splits **panes** vertically                                     |
|          `C-b <arrow key>`           | navigates split **panes**                                       |
|               `C-b z`                | fullscreen/minimizes active **pane**                            |
|           `C-d` OR `exit`            | closes currently active **pane**                                |
|               `C-b c`                | creates new **window**                                          |
|               `C-b p`                | switches to **previous window**                                 |
|               `C-b n`                | switches to **next window**                                     |
|               `C-b ,`                | renames current **window**                                      |
|               `C-b &`                | closes current **window**                                       |

Enters **pane resizing** mode by bringing up a prompt at the bottom of the pane
```console
C-b :

--Then type the following in the bottom prompt to  specify direction and # of cells to move  

resize-pane -<D, U, L, R> 10
```

