# Contribution Guide

!!! info "We value your contribution" 
    We are both a welcoming and an open community. 
    We warmly invite you to join us either as a user or a community contributor.
    We will be happy to hear from you.


**Please, follow these steps**

## Step 1: Forking and Installing IceVision

​1. Fork the repo to your own github account. click the Fork button to
create your own repo copy under your GitHub account. Once forked, you're
responsible for keeping your repo copy up-to-date with the upstream
icevision repo.

​2. Download a copy of your remote username/icevision repo to your
local machine. This is the working directory where you will make
changes:

```bash
git clone https://github.com/username/icevision.git
```

3.  Install the requirments. You many use miniconda or conda as well.

```bash
pip install -r requirements.txt
```

## Step 2: Stay in Sync with the original (upstream) repo

1.  Set the upstream to sync with this repo. This will keep you in sync
    with icevision easily.

```bash
git remote add upstream https://github.com/airctic/icevision.git
```

2.  Updating your local repo: Pull the upstream (original) repo.

```bash
git checkout master
git pull upstream master
```

## Step 3: Creating a new branch

```bash
git checkout -b feature-name
git branch
 master 
 * feature_name: 
```

## Step 4: Make changes, and commit your file changes

Edit files in your favorite editor, and format the code with
[black](https://black.readthedocs.io/en/stable/)

```bash
# View changes
git status  # See which files have changed
git diff    # See changes within files

git add path/to/file.md
git commit -m "Your meaningful commit message for the change."
```

Add more commits, if necessary.

## Step 5: Submitting a Pull Request

### A. Method 1: Using GitHub CLI

Preliminary step (done only once): Install gh by following the
instructions in [docs](https://cli.github.com/manual/installation) .

#### 1. Create a pull request using GitHub CLI

```bash
# Fill up the PR title and the body 
gh pr create -B master -b "enter body of PR here" -t "enter title"
```

#### 2. Confirm PR was created

You can confirm that your PR has been created by running the following
command, from the icevision folder:

```bash
gh pr list 
```

You can also check the status of your PR by running:

```bash
gh pr status 
```

More detailed documentation can be found
<https://cli.github.com/manual/gh_pr>.

#### 3. Updating a PR

If you want to change your code after a PR has been created, you can do
it by sending more commits to the same remote branch. For example:

```bash
git commit -m "updated the feature"
git push origin <enter-branch-name-same-as-before>
```

It will automatically show up in the PR on the github page. If these are
small changes they can be squashed together by the reviewer at the merge
time and appear as a single commit in the repository.

### B. Method 2: Using Git

#### 1. Create a pull request git

Upload your local branch to your remote GitHub repo
(github.com/username/icevision)

```bash
git push
```

After the push completes, a message may display a URL to automatically
submit a pull request to the upstream repo. If not, go to the
icevision main repo and GitHub will prompt you to create a pull
request.

#### 2. Confirm PR was created:

Ensure your pr is listed
[here](https://github.com/airctic/icevision/pulls)

3.  Updating a PR:

Same as before, normally push changes to your branch and the PR will get
automatically updated.

```bash
git commit -m "updated the feature"
git push origin <enter-branch-name-same-as-before>
```

* * * * *

## Reviewing Your PR

Maintainers and other contributors will review your pull request. Please
participate in the discussion and make the requested changes. When your
pull request is approved, it will be merged into the upstream
icevision repo.

> **note**
>
> IceVision has CI checking. It will automatically check your code
> for build as well.


## Feature Requests and questions

For Feature Requests and more questions raise a github
[issue](https://github.com/airctic/icevision/issues/). We will be happy
to assist you.

Be sure to check the
[documentation](https://airctic.github.io/icevision/index.html).

