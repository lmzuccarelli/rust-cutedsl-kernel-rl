use crate::config::load::WorkItem;
use crate::utils::common::get_item;
use custom_logger as log;
use std::process::Command;
use std::time::Instant;

pub trait ExecuteInterface {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct Execute {}

impl ExecuteInterface for Execute {
    // The work_item struct has the target_dir passed by the controller in the payload
    // It's not set via the get_item utility function as its changes
    // for each trajectory_xxx/step_x
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing kernel code");
        let start = Instant::now();
        let cutedsl_env_path = get_item("cutedsl_env_path")?;

        let kernel_file = match work_item.kernel_file {
            Some(name) => name,
            None => {
                return Err(Box::from(
                    "[run] (executeinfterface) kernel_file field missing",
                ));
            }
        };

        log::debug!(
            "[run] (executeinterface) target directory {}",
            work_item.target_dir
        );
        log::debug!("[run] (executeinterface) kernel {}", kernel_file);

        // this is important as we need the exact path to the cutlass directory
        // with all dependencies set (i.e flashinfer etc)
        let output_res = Command::new(format!("{}/bin/python", cutedsl_env_path))
            .args(vec![format!("{}/{}", work_item.target_dir, kernel_file)])
            .output();

        let response = match output_res {
            Ok(result) => {
                let stdout = String::from_utf8_lossy(&result.stdout).trim().to_string();
                let stderr = String::from_utf8_lossy(&result.stderr).trim().to_string();
                if !result.status.success() || stdout.to_lowercase().contains("failed") {
                    let output = match stderr.is_empty() {
                        true => stdout,
                        false => stderr,
                    };
                    return Err(Box::from(format!(
                        "[run] (executeinterface) result (exited non zero) {}",
                        output.to_string()
                    )));
                }
                // preserve output
                println!("{}", stdout);
                stdout
            }
            Err(e) => {
                log::error!("[run] (executeinfterface) {}", e);
                return Err(Box::from(format!("[run] (executeinterface) {}", e)));
            }
        };

        let elapsed = start.elapsed();
        log::info!("[run] (executeinterface) : completed task in {:?}", elapsed);

        Ok(response)
    }
}
