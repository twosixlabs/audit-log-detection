# Malicious Behavior Detection using Windows Audit Logs, Part II

This is the github page for the public release of a dataset of Windows Audit Logs and the associated publication.

The free pre-print of the publication will be located at http://arxiv.org, once it is availiable. 

__If you have any questions or issues, or something is not clear about our data or scripts, please report them, so we can fix them as soon as possible.__

###Synoposis

This project is a contininioution of the previous projected on behavior malware detection using audit log. The github for the previous project can be found [here](https://github.com/konstantinberlin/malware-windows-audit-log-detection).

Similar to the previous study, we continued to investigate the utility of agentless detection of malicious endpoint behavior, using only the standard build-in Windows audit logging facility as our signal. In the process of performing the study we collected a much larger and diverse set of audit logs, from sandboxed CuckooBox runs, as well as directly from our enterprise network.

We would like to encourage research in the security critical area of behavior malware detection by releasing this (annonymized) dataset to the public. Along with the dataset, once availiable, we will post detection benchmarks (including the code), which we encourage the community to use as reference in their publications.

## Data and Anonymization 

The anonymized version of the data that we used to compute our results will be found [here (__link currently broken__) ](???). This data will be anonymized in order to protect the privacy of the users from which it was collected, but with the second goal to allow the security community to supplement and reuse our data for their own needs. The process by which this was done is described below.

The following is the description of the anonymization steps:

1. Transform all the entries using the regex transformations defined in [].
2. Observe all the paths (including directory and name) and the subpaths for files, process names, and registry entries, for all the CuckooBox derived Windows audit logs (ex., c:\, c:\windows, c:\window\system32, etc). Put them in a bag of public paths called __P__.
3. Encrypt the file/registry/process names in pieces.
    * Do not encrypt any logs from the CuckooBox runs.
    * For each audit log entry in the enterprise data, see if all or parts of its path is in __P__. Encrypt each directory/registry using its name, if the full path of the directory/registry is not in __P__, otherwise leave unencrypted. The encrypted name is the text `sha1_` followed by the sha1 of the directory/registry name. For files we leave the extension exposed, and only hash the name. Ex. `[windows]\system32\fake_dir\fake.dll` will be encrypted as `[windows]\system32\sha1_<hash of "fake_dir">\sha1_<hash of "fake">.dll`.
    * For sensitive files types, like documents, slides, text, etc., we salt the filenames before hashing.

### Regex Transformations

The regex transformations that we use to generate our feature labels are located in the [regex file](regex.txt). The regex expressions must be executed in order listed to reproduce our results.

### Data Content

The following is the data that we used for our analysis. The file format is specified in Section [File Formats](#ff).

#### Overview

The root directory contains the following:

* `auditlog_meta.db` - the metadata Sqlite3 database containing labels and family names for the audit logs.
* `auditlog_matrix_rocksdb` - the RocksDB file containing the 3-gram features for each log. Stored as sparse binary double array, in (index,value) touples. Below is example Python code that loads the feature matrix from this file assuming a list of log names `names_all`.
```python
def matrix_load_rocksdb(rocks_db, batch_idx, names_all):

    for i in xrange(0, len(batch_idx), 1):
        
        #get the files
        local_names = [names_all[x] for x in batch_idx[i:(i+1)]]
        
        entry_vals = rocks_db.multi_get(local_names)
        
        for name in local_names:
            val = entry_vals[name]
            vals = np.array(struct.unpack(">{0}d".format(int(len(val)/8)), val))
            
            #append to storage
            A.append(vals)

    return A
```
* `auditlog_features_rocksdb` - the RocksDB file containing the 3-gram feature names, as used in the touple index above.

* `auditlog_json_rocksdb` - the JSON formatted and zlib compressed "raw" log that was used to generated all the data. Example Python code to read and print one entry is below:
```python
   #example code to read one entry
   db = rocksdb.DB(os.path.join(path,'auditlog_json_rocksdb'), rocksdb.Options(create_if_missing=False), read_only=True)
   it = db.iteritems()
   it.seek_to_first()
   for name,v in it:
      v = zlib.decompress(v, 16+zlib.MAX_WBITS)
      print name, v
      break  
```

## Copyright and License

Code, documentation, and data copyright 2015-2016 Invincea Labs, LLC. Release is governed by [Apache 2.0](LICENSE.txt)  license.

