# SHELL SNIPPETS <!-- omit in toc -->

- [BASH](#bash)
  - [Navigation Commands](#navigation-commands)
  - [Modifier Commands](#modifier-commands)
- [GIT](#git)
  - [Local Repository Setup / Update](#local-repository-setup--update)
  - [Update Main Repository](#update-main-repository)
  - [Branch Repository](#branch-repository)
- [CONDA](#conda)
- [UNIX](#unix)
  - [SETUP OF UNIX USER](#setup-of-unix-user)
  - [SETUP OF NGINX ON SERVER](#setup-of-nginx-on-server)
- [TMUX](#tmux)
- [POSTGRESQL](#postgresql)

## BASH

---

### Navigation Commands

|      Commands       |                                                                                                 |
| :-----------------: | ----------------------------------------------------------------------------------------------- |
|        `pwd`        | Displays the **present working directory**, or the path to the current directory                |
|        `ls`         | **Lists** all non-hidden items/documents in the current directory; add '-A' flag to show hidden |
|  `cd folder_name`   | Used to **change directory**, or to navigate into the directory/folder specified                |
|      `history`      | Provides the **historical** commands passed through the Shell                                   |
| `man shell_command` | Provides the **manual** for the shell command specified                                         |
|  ` cat file-name`   | Displays the first few lines of a file in the shell                                             |

### Modifier Commands

|                Commands                |                                                                                  |
| :------------------------------------: | -------------------------------------------------------------------------------- |
|      `touch file_name.file_type`       | **Creates a file** in the current location/directory, in the file type specified |
|         `mkdir new_directory`          | creates a new directory in current location                                      |
|             `rm file_name`             | **Deletes the file** specified (cannot delete a folder/directory)                |
|          `rm -rf folder_name`          | **Deletes an entire folder/directory** and its contents                          |
|   `cp file_name copy/location/path`    | copies file_name to the directory specified                                      |
|    `mv file_name new/location/path`    | moves file_name to the directory specified                                       |
| `cp -R folder_name copy/location/path` | copies folder_name to the directory specified                                    |
| `mv -R folder_name new/location/path`  | moves folder_name to the directory specified                                     |
|          `echo hello world!`           | displays or prints the variable or string specified                              |

`grep` searches `file_name` for `search_term`

```console
grep search_term file_name

# FLAGS
-c : count of matching lines, not matching text
-h : do NOT print names of files when searching multiple files
-i : ignore case sensitivity
-l : print names of matching files, not matching text
-n : print line numbers for matching lines
-v : invert the match; show lines that DON'T match
```

`wc` is a word count that displays the number of character, words, or lines in `file_name`

```console
wc file_name

# FLAGS
-c : counts characters in file
-w : counts words in file
-l : counts lines in file
```

## GIT

---

### Local Repository Setup / Update

|                   Commands                   |                                                                                                              |
| :------------------------------------------: | ------------------------------------------------------------------------------------------------------------ |
|                  `git init`                  | initializes the current repository to be tracked by Git                                                      |
| `git clone "URL-OF-REPOSITORY-TO-BE-CLONED"` | clones the repository using the respository URL to the local machine                                         |
|                 `git stash`                  | stashes local repository file/folder changes before running `git pull`                                       |
|                  `git pull`                  | updates local repository with the main repository                                                            |
|               `git stash pop`                | after a `git pull`, reapplies stashed local repository file/folder changes                                   |
|              `git reset --hard`              | reset the local repository to the last instance of the main repository, losing all local file/folder changes |

### Update Main Repository

|             Commands             |                                                                                        |
| :------------------------------: | -------------------------------------------------------------------------------------- |
|           `git status`           | displays the status of the local repository, tracked/untracked files, etc.             |
| `git add file_name folder_name`  | stages untracked files/folders to be uploaded to the main repository                   |
|   `git rm --cached file_name`    | unstages files/folder previously staged by running `git add`                           |
| `git commit -m "commit message"` | commits files/folders by running `git add`to the main repository w/accompaning message |
|        `git rm file_name`        | removes files/folders previously committed by running `git commit`                     |
|            `git push`            | pushes the committed files/folders to the main repository                              |

### Branch Repository

|     Commands     |                                                      |
| :--------------: | ---------------------------------------------------- |
| `git branch -a ` | displays any branched repository for the Git project |

Creates a merge request by checking out a branch repository.

```console
git checkout -b my-new-branch

gid add.
git commit -m 'My branch merge request info.'

git push origin my-new-branch

# GitLab prompts you with a direct link for creating a merge request, copy/paste link into broswer to complete
```

## CONDA

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

## UNIX

---

|                     Commands                     |                                                                               |
| :----------------------------------------------: | ----------------------------------------------------------------------------- |
|                   `uname -srm`                   | display which version of the Linux kernel is running.                         |
| `uname --kernel-name --kernel-release --machine` | provide a longer, more descriptive version.                                   |
|              `cat /etc/os-release`               | display information on what distribution is running on the system.            |
|               `adduser user-name`                | creates a user on the system that can be set with non-root permissions.       |
|                    `sudo su`                     | switches from current user configuration to the **root** user configuration.  |
|                 `apt-get update`                 | gets updates for the server, with applying them; add `sudo` to apply changes. |
|             `apt-get <package-name>`             | install a single package to the server.                                       |
|                `sudo ufw status`                 | check status of the Ubuntu firewall for the server.                           |
|                `sudo ufw enable`                 | enables the firewall on the server.                                           |
|           `systemctl status <engine>`            | provides system level information on the engine passed.                       |
|     `systemctl start/stop/restart <engine>`      | starts, stops, or resets the engine from systemctl.                           |

creates a **log of the output** of a .py file to the file specified

```console
ipython python_file.py prod 2>&1 | tee log/output_file.txt

# ipython/python can be used
# 'prod 2>&1' indicates to record "standard errors" (2) in the same location as "standard output" (1)
# 'tee' displays output and records it to log/output_file.txt
```

### SETUP OF UNIX USER

allows the specified user to **temporarily** gain access to ROOT user privledges.

```console
# run 'visudo' command to access file.
visudo

# add user under 'User privilege specification' column matching ROOT credentials
# save file and exit
```

allows the user to login directly to the UNIX server

```console
# run to access config file
vi /etc/ssh/sshd_config

# confirm Authentication --> PermitRootLogin = no
# scroll to end of document and insert 'AllowUsers <username>'
# enter ':wq' to write to file and quit

# reloads the service to accept the new cofguration
service sshd reload
```

### SETUP OF NGINX ON SERVER

allows nginx to bypass the firewall

```console
# enter root user
sudo su

# enables firewall on server if not yet enabled.
sudo ufw enable

# adds nginx to access the server through the firewall
sudo ufw allow 'Nginx HTTP'

# setup nginx config file to accept REST API
sudo vi /etc/nginx/sites-available/items-rest.conf

# create symlink to config file to the 'sites-enabled' folder where nginx reads config properties
sudo ln -s /etc/nginx/sites-available/items-rest.conf /etc/nginx/sites-enabled

# create directory for the app
sudo mkdir /var/www/html/items-rest
```

## TMUX

---

|               Commands               | `C-b==CTRL+b` `C-d==CTRL+d`                                     |
| :----------------------------------: | --------------------------------------------------------------- |
|      `tmux new -s session-name`      | creates new session named `session-name`                        |
|              `tmux ls`               | displays all **sessions** currently running (detached/attached) |
|          `C-b d` or `C-b D`          | detaches current **session** _(D allows selection)_             |
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

## POSTGRESQL

---

|  Commands   |                                                                           |
| :---------: | ------------------------------------------------------------------------- |
|   `psql`    | connects to the PostgreSQL shell.                                         |
| `\conninfo` | displays the database, user, socket, and port information for connection. |
|    `\q`     | leaves the PostgresSQL sheel.                                             |

instead of postgresSQL accepting connection via same name of user/database, the commands below will make postgresSQL request the user password. Allows SQLAlchemy to preform correctly.

```console
# navigates to the postgresSQL security file as ROOT user
sudo vi /etc/postgresql/<version>/main/pg_hba.conf

# confirm the "'local' is for Unix domain socket connections only" METHOD value is set to 'md5' not 'peer'
```

create UNIX user, in postgres user (created when postgres installed) with PostgreSQL permissions

```console
# login as postgres user from root/user
sudo -i -u postgres

# create postgresSQL user w/same name as user profile and enter password
createuser <username> -P

# creates database for the user
createdb username

# by default, connnecting w/'psql' connects to database named the same as the user
```
