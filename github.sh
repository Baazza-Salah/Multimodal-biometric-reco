#!/bin/bash

# 2. Remove the old Git history
rm -rf .git

# 3. Initialize a new Git repository
git init

# 4. Add all files
git add .

# 5. Commit with a new message
git commit -m "Initial commit for new repository"

# 6. Add your new remote repository
# Replace the URL below with your new GitHub repo URL
git remote add origin https://github.com/Baazza-Salah/Multimodal-biometric-reco

# 7. Push to the new repository (main branch)
git branch -M main
git push -u origin main

echo "Repository is now independent and pushed to the new remote!"
