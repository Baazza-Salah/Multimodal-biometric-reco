#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Git History Reset Script${NC}\n"
echo -e "${YELLOW}This will remove all git history and create a fresh repository.${NC}"
echo -e "${RED}WARNING: This action cannot be undone!${NC}\n"

# Confirm action
echo -e "${YELLOW}Are you sure you want to proceed? (yes/no):${NC}"
read -r CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo -e "${RED}Operation cancelled.${NC}"
    exit 0
fi

echo -e "\n${YELLOW}Step 1: Removing .git directory...${NC}"
rm -rf .git

echo -e "${GREEN}✓ Git history removed${NC}\n"

echo -e "${YELLOW}Step 2: Initializing new git repository...${NC}"
git init

echo -e "${GREEN}✓ New repository initialized${NC}\n"

echo -e "${YELLOW}Step 3: Adding all files...${NC}"
git add .

echo -e "${GREEN}✓ Files staged${NC}\n"

echo -e "${YELLOW}Step 4: Creating initial commit...${NC}"
git commit -m "Initial commit"

echo -e "${GREEN}✓ Initial commit created${NC}\n"

echo -e "${YELLOW}Step 5: Renaming branch to main...${NC}"
git branch -M main

echo -e "${GREEN}✓ Branch renamed to main${NC}\n"

echo -e "${YELLOW}Step 6: Adding remote origin...${NC}"
git remote add origin https://github.com/Baazza-Salah/Multimodal-biometic

echo -e "${GREEN}✓ Remote origin added${NC}\n"

echo -e "${YELLOW}Step 7: Force pushing to GitHub...${NC}"
echo -e "${RED}This will overwrite the existing repository!${NC}"
git push -u origin main --force

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Successfully reset and pushed!${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    echo -e "Repository URL: ${GREEN}https://github.com/Baazza-Salah/Multimodal-biometic${NC}\n"
    echo -e "${GREEN}✓ All previous history removed${NC}"
    echo -e "${GREEN}✓ Only your commit exists now${NC}\n"
else
    echo -e "${RED}Failed to push to GitHub!${NC}"
    exit 1
fi
