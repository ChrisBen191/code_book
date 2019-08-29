# TERMINAL NAVIGATION COMMANDS

Used to **change directory**, or to navigate into the directory/folder specified

        $ cd folder_name

Displays the **present working directory**, or the path to the current directory

        $ pwd

**Lists** all non-hidden items/documents in the current directory

        $ ls

**Lists** all hidden and non-hidden items in the current directory

        $ ls -A

Creates a file in the current location, in the file type specified 

        $ touch file_name.file_type	

Copies the specified file to the specified directory

        $ cp file_to_be_moved    directory/path/to/copy/file

Moves the specified file to the directory specified

        $ mv file_to_be_moved    directory/path/to/move/file

Deletes the file (cannot delete a folder/directory)

        $ rm file_name

Creates a new directory in the current directory

        $ mkdir folder_name	

Copies the specified folder to the specified directory

        $ cp -R folder_to_be_moved    directory/path/to/move/folder

Deletes an entire folder/directory and its contents

        $ rm -rf folder_name	

# GITHUB COMMANDS

Clones the specified respository to the local machine

        $ git clone "URL-OF-REPOSITORY-TO-BE-CLONED"

Shows the status of the current local repository; will show up-to-date, behind master, etc.

        $ git status

Stages all unstaged files to be uploaded

        $ git add -A

Commits the staged files to be pushed to remote repository; requires message

        $ git commit -m "put your message here(usually what was added)"	

Pushes the committed files to remote repository

        $ git push	

Resets the local machine to last instance pulled from master. **All local changes will be lost.**

        $ git reset --hard

Stashes local changes to the repository before performing a 'git pull'. **Local changes will not be lost.**

        $ git stash	

Updates local machine from a remote repository
        
        $ git pull

Performed after a **git pull**; reapplies stashed local files and changes to the repository

        $ git stash pop	



