# GitHub Deployment Guide

This document provides detailed step-by-step instructions for deploying your ML Bootcamp project to GitHub.

## Prerequisites

Before you start, make sure you have:
- A GitHub account
- Git installed on your local machine
- Your ML Bootcamp project files ready

## Step 1: Initialize Local Git Repository

If you haven't already, navigate to your project directory and initialize git:

```bash
cd ml-bootcamp
git init
```

## Step 2: Configure Git

Set your name and email if you haven't already:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Add Files and Make Initial Commit

Add all your project files to git:

```bash
git add .
git commit -m "Initial commit: Complete ML Bootcamp project"
```

## Step 4: Create GitHub Repository

1. Open your web browser and go to https://github.com/new
2. Enter a repository name (e.g., "ml-bootcamp")
3. Choose Public or Private visibility
4. **Important**: Do NOT initialize with README, .gitignore, or license since you already have them
5. Click "Create repository"

## Step 5: Link Local and Remote Repositories

Copy the repository URL from your new GitHub repository page. It will look like:
`https://github.com/YOUR_USERNAME/ml-bootcamp.git`

Add it as a remote:

```bash
git remote add origin https://github.com/YOUR_USERNAME/ml-bootcamp.git
```

## Step 6: Push to GitHub

```bash
git branch -M main
git push -u origin main
```

If you're prompted for credentials:
- Use your GitHub username and password
- Or for more secure access, create a Personal Access Token at 
  https://github.com/settings/tokens and use that as your password

## Step 7: Verify Deployment

After pushing, check your GitHub repository page. You should see:
- All your files in the repository
- Your README.md displayed on the main page
- The correct folder structure

## Step 8: Set Up GitHub Pages (Optional)

To deploy your project documentation as a website:

1. Go to your repository on GitHub
2. Click on "Settings"
3. Scroll down to the "GitHub Pages" section
4. Select the "main" branch as the source
5. Click "Save"
6. Your site will be available at `https://YOUR_USERNAME.github.io/ml-bootcamp`

## Step 9: Create a Requirements File for Reproducibility

If you haven't already, make sure your requirements.txt file is up-to-date:

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push
```

## Step 10: Branching Strategy

For ongoing development, follow this branching workflow:

```bash
# Create a new feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Commit your changes
git add .
git commit -m "Add your feature"

# Push branch to GitHub
git push origin feature/your-feature-name

# On GitHub, create a pull request to merge into main
```

## Step 11: Adding Collaborators (If Needed)

1. Go to your repository on GitHub
2. Click on "Settings"
3. Click on "Collaborators and teams"
4. Click "Add people"
5. Enter the GitHub username of the person you want to add

## Step 12: Protecting Your Main Branch (Optional but Recommended)

1. Go to your repository
2. Click on "Settings"
3. Click on "Branches"
4. Click "Add rule"
5. Type "main" as the branch name pattern
6. Configure the protection rules (e.g., require pull request reviews)
7. Click "Save changes"

## Step 13: Project Management

Consider adding a PROJECT.md file to track your roadmap:

1. Go to your repository
2. Click on "Issues"
3. Click on "New issue"
4. Use the "Projects" tab to create a kanban board

## Step 14: Security Considerations

- Never commit sensitive data (API keys, passwords)
- Use environment variables for configuration
- Use .gitignore to exclude sensitive files
- Consider using GitHub secret scanning for public repos

## Step 15: Continuous Integration (Optional)

Set up GitHub Actions for automated testing:

1. Create `.github/workflows/test.yml`
2. Add configuration for running tests
3. Commit and push this file
4. GitHub will automatically run tests on each push

## Troubleshooting

### Authentication Error
If you get an authentication error:
1. Create a Personal Access Token at https://github.com/settings/tokens
2. Use the token as your password when prompted
3. On Windows, you may need to use a credential manager

### Push Rejected
If your push is rejected:
```bash
git pull origin main
git push origin main
```

### Large Files
If you're pushing files larger than 100MB:
1. Use Git LFS (Large File Storage)
2. Or store large files elsewhere (e.g., cloud storage, release assets)

Congratulations! Your ML Bootcamp project is now on GitHub and ready for collaboration.