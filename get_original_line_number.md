# algorithm to trace line number changes through a git tree

We are given the following:

* A git repository (`git_repo`)
* The name of a file in which the change took place (`file_name`)
* The commit in which the change was made (`changing_commit`)
* The original commit identified with `git blame` (`original_commit`)
* The line number in which the changes took place (`post_line_num`)
* Aliases of the file name in case the file was remed or moved in the past (`aliases`)

We now aim to identify the original line number of the line that was authored in `original commit` and modified in `changing_commit`.

We use the following procedure:
1. We initialise a commit DAG linking `changing_commit` to its parents.
2. We then expand the commit DAG linking all leaf nodes in the DAG to their reppective parents. We continue expanding as long as
   1. `original_commit` is not yet in the commit tree,
   2. There is more than one leaf node in the tree. This condition is important as only when both conditions are satisfied we can be certain (due to the properties of a DAG) that there cannot paths from `changing_commit` to `original_commit` that are not shown in the DAG.
3. We identify all paths from `changing_commit` to `original_commit`. Due to working on a finite DAG we know that there is a finite number of finite paths. Due to the process with which the DAG was created, we know that at least one path exists.
4. We now trace back the line numbers through the commits on the extracted paths knowing that only one path can be the correct one. While tracing back we check if the line exists on the correct position in the file. Note, that we only need to trace the origin of lines that were deleted or modified, as additions do not have another commit as their origin. Depending on if the `changing_commit` is a `merge` or not, the following cases need to be considered:
   1. If the `changing_commit` is not a `merge` we can be certain that the change is valid.
      1. The line no longer exists in the file but we have not yet reached the `original_commit`. This means we are in on the wrong path. We drop the path and try the next one.
      2. We arrive at a file that reports
   2. If the `changing_commit` is a `merge`. In this case we cannot be sure if the change was made together with the `merge` or if a change made before the merge is recorded in the merge.