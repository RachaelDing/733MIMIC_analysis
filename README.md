Create table patients (
  subject_id INT,
  dob TIMESTAMP,
  PRIMARY KEY (subject_id)
);

Create table admissions (
  subject_id INT,
  hadm_id INT,
  admittime TIMESTAMP,
  admission_type TEXT,
  PRIMARY KEY (hadm_id)
);

Create table temp51006 (
  row_id INT,
  subject_id INT,
  hadm_id INT,
  charttime TIMESTAMP,
  valuenum DOUBLE,
  valueuom TEXT,
  PRIMARY KEY (row_id)
);

Create table item51006 (
  subject_id INT,
  hadm_id INT,
  charttime TIMESTAMP,
  valuenum DOUBLE,
  PRIMARY KEY (hadm_id, subject_id)
);


Create table chartevent (
  row_id INT,
  subject_id INT,
  hadm_id INT,
  icustay_id INT,
  itemid INT,
  charttime TIMESTAMP,
  storetime TIMESTAMP,
  cgid INT,
  value TEXT,
  valuenum FLOAT,
  valueuom TEXT,
  warning INT,
  error INT,
  resultstatus TEXT,
  stopped TEXT,
  PRIMARY KEY (row_id)
) ;

