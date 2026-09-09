use crate::config::load::WorkItem;
use custom_logger as log;
use std::env;
use std::fs;

pub trait FileInterface {
    async fn kernel_rw(
        work_item: WorkItem,
        write: bool,
    ) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct FileOperation {}

impl FileInterface for FileOperation {
    async fn kernel_rw(
        work_item: WorkItem,
        write: bool,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // restore working dir
        env::set_current_dir(work_item.working_dir)?;

        let mut kernel_code = String::new();
        let dir = work_item.target_dir.clone();
        log::debug!("[cuda_kernel_rw] directory {}", dir);
        // create output directory (incase its not created)
        fs::create_dir_all(format!("{}/build", dir))?;

        match work_item.kernel_name {
            Some(name) => {
                let file = format!("{}/{}", dir, name);
                log::debug!("[cuda_kernel_rw] file {}", file);
                log::trace!("[cuda_kernel_rw] kernel_code {}", kernel_code);
                if write && work_item.code.is_some() {
                    kernel_code = work_item.code.unwrap_or("".to_string());
                    fs::write(file, kernel_code.clone())?;
                } else {
                    kernel_code = fs::read_to_string(&file)?;
                }
            }
            None => {
                let file = format!("kernel-cutedsl/{}", work_item.name);
                kernel_code = fs::read_to_string(&file)?;
            }
        }
        Ok(kernel_code)
    }
}
