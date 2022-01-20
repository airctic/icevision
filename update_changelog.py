import argparse
from io import StringIO
import json
from github import Github


def get_closed_pull_requests():
    icevision_repo = g.get_repo("airctic/icevision")
    pull_requests = icevision_repo.get_pulls(state="closed")
    return pull_requests


def filter_pull_requests(
    pull_requests,
    start_pull_request_id=None,
    end_pull_request_id=1,
    end_is_off_by_one=False,
):
    "Returns all pull requests that have the same pull request id as the given limits or are inbetween."
    collected_pull_requests = []
    page_counter = 0
    if start_pull_request_id is None:
        collect_flag = True
    else:
        collect_flag = False
    stop_flag = False
    while True:
        try:
            # pull requests will are sorted by newest
            pull_request_page = pull_requests.get_page(page_counter)
            # stop when a empty page appears as only empty pages follow onwards
            if len(pull_request_page) == 0:
                break
            for pull_request in pull_requests:
                if start_pull_request_id == pull_request.number:
                    collect_flag = True
                if collect_flag:
                    collected_pull_requests.append(pull_request)
                if end_pull_request_id == pull_request.number:
                    stop_flag = True
                    # if the end is off by one due to using the end of the version before the last element of the
                    # collected pull requests needs to be removed0
                    if end_is_off_by_one:
                        collected_pull_requests = collected_pull_requests[:-1]
                    break
            page_counter += 1
            if stop_flag:
                break
        except:
            break
    return collected_pull_requests


def extract_contribution_data_from_pull_requests(pull_requests):
    data = {}
    for pull_request in pull_requests:
        if not pull_request.user.login in data.keys():
            data[pull_request.user.login] = [
                {
                    "title": pull_request.title,
                    "id": pull_request.number,
                    "url": pull_request.comments_url,
                }
            ]
        else:
            data[pull_request.user.login].append(
                {
                    "title": pull_request.title,
                    "id": pull_request.number,
                    "url": pull_request.comments_url,
                }
            )
    return data


0


def clean_comment_url(url):
    clean_url = url.replace("//api.github", "//github")
    clean_url = clean_url.replace("/repos/", "/")
    clean_url = clean_url.replace("/comments", "")
    return clean_url


def create_markdown_from_pull_request_data(data, version_name=None):
    version_name = "VERSION" if version_name is None else version_name
    markdown = f"""[comment]: # (Version_start)
## {version_name}\nThe following PRs have been merged since the last version.\n\n"""

    # add contributers with their contributions sorted by most contributions
    sorted_data = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)
    for author, pull_requests in sorted_data:
        markdown += author + "\n"
        for pull_request in pull_requests:
            markdown += f"  - [{pull_request['title']}]({clean_comment_url(pull_request['url'])}) (#{pull_request['id']})\n"
        markdown += "\n"
    # add thank you notice
    markdown += (
        "**Thank you to all contributers: "
        + ", ".join([f"@{entry[0]}" for entry in sorted_data])
        + "**"
        + "\n\n[comment]: # (Version_end)\n\n"
    )
    return markdown


def load_loged_data():
    try:
        data = json.load(open("logs.json", "r"))
    except:
        data = {}
    return data


def log_update(log_data, version_string, pull_requests_data):
    log_data[version_string] = pull_requests_data
    log_data["latest_release"] = pull_requests_data
    json.dump(log_data, open("logs.json", "w"))


def get_latest_pull_request_id(log_data):
    pull_request_ids = []
    for contributer_pull_requests in log_data["latest_release"].values():
        pull_request_ids += [
            pull_request["id"] for pull_request in contributer_pull_requests
        ]
    return max(pull_request_ids)


def load_changelogs():
    change_log = open("CHANGELOG.md", "r").read()
    with open("CHANGELOG_backup.md", "w") as f:
        f.write(change_log)
    changelogs_head, changelog_body = change_log.split(
        "[comment]: # (Add changes below)"
    )
    return changelogs_head, changelog_body


def update_changelog_file(markdown_update_text):
    changelogs_head, changelog_body = load_changelogs()
    new_changelog_file = """"""
    new_changelog_file = changelogs_head
    new_changelog_file += "[comment]: # (Add changes below)\n\n"
    new_changelog_file += markdown_update_text
    new_changelog_file += changelog_body
    with open("CHANGELOG.md", "w") as f:
        f.write(new_changelog_file)


parser = argparse.ArgumentParser(
    description="""Automatically update the CHANGELOG.md file. The script can create an update between two given PRs, were all RPs including the start and end PR are listed for the new version. Or only one can be provided. 
If only the start PR is provided all PRs after wards will be used. When only the end PR is provided the PR after the one that is the end PR of the last version will be used as the start PR. If both are not given both methods described before will be combined to 
get the start and end PRs."""
)
parser.add_argument(
    "github_token", metavar="T", type=str, help="Access token for github."
)
parser.add_argument(
    "version", metavar="V", type=str, help="String for the name of the next version"
)
parser.add_argument(
    "--start_pr",
    default=None,
    type=int,
    help="Id of the pr to start the collection of PRs looking backward in time. This should be a bigger than the value of --end_pr.",
)
parser.add_argument(
    "--end_pr",
    default=None,
    type=int,
    help="Id of the pr to end the collection of PRslooking backward in time. This sould be smaller than the value of --start_pr",
)

if __name__ == "__main__":

    args = parser.parse_args()

    # load changelog data
    print("Loading changelog data.")
    changelog_data = load_loged_data()
    end_pull_request_id_is_off_by_one = False
    if args.start_pr is None:
        start_pull_request_id = get_latest_pull_request_id(changelog_data)
        end_pull_request_id_is_off_by_one = True
    else:
        start_pull_request_id = args.start_pr
    print(f"Starting from PR: {start_pull_request_id}")

    # generate markdown for new release
    print(f"Generating markdown for new version: {args.version}")
    g = Github(args.github_token)
    pull_requests = get_closed_pull_requests()
    selected_pull_requests = filter_pull_requests(
        pull_requests,
        end_pull_request_id=start_pull_request_id,
        start_pull_request_id=args.end_pr,
        end_is_off_by_one=end_pull_request_id_is_off_by_one,
    )
    pull_requests_data = extract_contribution_data_from_pull_requests(
        selected_pull_requests
    )
    markdown_update = create_markdown_from_pull_request_data(
        pull_requests_data, args.version
    )

    # apply updates
    print("Updating logs and markdown.")
    log_update(changelog_data, args.version, pull_requests_data)
    update_changelog_file(
        create_markdown_from_pull_request_data(pull_requests_data, args.version)
    )

    print("Done!")
