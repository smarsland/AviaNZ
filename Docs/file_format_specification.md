# Specification of file formats used by AviaNZ

AviaNZ annotations and filter definitions are stored in JSON format to allow easy parsing an manual inspection by text editors.

## Annotation files (.data)
A JSON array where the first (optional, but recommended) element stores metadata about the corresponding audio file, and each remaining element corresponds to a segment:

    [ Meta, seg, seg, seg, seg ... ]

`Meta`: a JSON object (key-value pairs) containing any metadata. Required fields:  
`Operator` - string  
`Reviewer` - string  
`Duration` - numeric, audio file length, in seconds  
...

Each true segment `seg` is a JSON array containing five elements, all required:

    [ starttime, endtime, freq.low, freq.high, labels ]
    
`startime, endtime` - segment start and end positions, in seconds, relative to start of file as 0.  
`freq.low, freq.high` - for annotation boxes, frequency band in Hz. For segments (full-band annotations), both `0`. If both `0<freq<1`, old format is assumed, and treated as full-band segment (`0,0`).  
`labels` - a JSON array of labels for each type of sound detected:

    [ label, label, label... ]
    
where each `label` is a JSON object, having some of the following fields:

    { "species": "Kiwi (Little spotted)", "certainty": 0, "filter": "kiwi-best", "calltype": "f1", ... }
    
`species` - string, either `"genus (species)"` or just plain `"species"`. May be `"Don't Know"` or any other label (`"Bellbird/Tui"`, `"Fantail (spp)"`...), except for the internal genus separator `>`. Required.  
`certainty` - numeric between 0 and 100. Currently, for `"species": "Don't Know"` only `0` allowed, `100` corresponds to green segments, and `50` corresponds to question marks in earlier formats. `(species, certainty)` defines a unique key for labels. Required.  
`filter` - string, name of the filter file that created this label, or `"M"` for manual annotations.  
`calltype` - string, to identify the call type. Call types can be annotated manually, or will be automatically generated from clusters during filter training. Required for automatic filters (i.e. if `filter` is not empty or `"M"`).  
Any additional attributes defined for this call (male/female, subjective loudness...) are optional and can be passed as key-value pairs.

Thus, a full .data file may look like this:

    [ {"Operator": Alice, "Reviewer": Bob, "Duration": 60.0, "Noise": "windy"},    // metadata
      // a manually marked box
      [1.0, 19.0, 1200, 2500,
        [
          { "species": "Kiwi (Little spotted)", "certainty": 100, "filter": "M", "loudness": 3 }
        ]
      ],
      // box from a "trill" filter
      [21.0, 23.0, 800, 6000,
        [
          { "species": "Morepork", "certainty": 50, "filter": "ruru-90-10", "calltype": "trill" }
        ]
      ],
      // a manually marked segment with morepork and something else
      [35, 45, 0, 0,
        [
          { "species": "Morepork", "certainty": 100, "filter": "M" },
          { "species": "Don't Know", "certainty": 0, "filter": "M" }
        ]
      ]
    ]


## Filter files (.txt)

A JSON array:

    { "species": "Kiwi (Little spotted)", "SampleRate": 16000, "Filters": [], ...}
    
Main filter ID is the file name because this automatically ensures that no duplicate IDs are present at any installation of AviaNZ. This name can be any string permitted by the OS, and no further information is gathered from it.  
`species` - string. This label will be assigned as the `species` in segments generated by this filter. Can follow `"genus (species)"` format as described above. Required.  
`SampleRate` - integer. All analyses will be done after down-(up-)sampling to this rate. Required.  
Any extra parameters to be applied for all subfilters may be provided (such as `"wind"`).

`Filters` - JSON array of filters corresponding to each type of call (at least one element). Each is a JSON object:

    { "calltype": "clust1", "WaveletParams": {"thr": 0.5, "M": 1.5, "nodes": [35, 37, 40]}, "FreqRange": [1000, 3000], ... }
    
`calltype` - either user-defined call type, or automatically generated cluster ID. String. Required.  
`WaveletParams` - JSON object of parameters needed for wavelet filtering. Required. Currently needs:
`thr` - numeric, threshold for detecting calls. Required.  
`M` - numeric, energy curve window in seconds. Required.  
`nodes` - JSON array of wavelet nodes used in this filter. Required.  
`FreqRange` - frequency band for analysis. Identified calls will be marked as boxes with these limits, or as full-band segments if not provided.
Any extra subfilter parameters may follow, such as `"F0"`.

Thus, a full filter file may look like this:

    { "species": "Kiwi (Little spotted)", "SampleRate": 16000, "Rain": false, "Wind": true,
      "Filters": [
        { "calltype": "M", "WaveletParams": {"nodes": [44, 45, 46], "thr": 0.5, "M": 1.5}, "F0": true, "FreqRange": [1500, 5000] },
        { "calltype": "F", "WaveletParams": {"nodes": [41, 44], "thr": 0.8, "M": 2}, "FreqRange": [1000, 2500] }
      ]
    }
